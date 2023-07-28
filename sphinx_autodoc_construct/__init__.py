import os
import inspect
import importlib
from contextlib import contextmanager, ExitStack
from docutils import nodes
from pathlib import Path
import json
import sphinx
import construct
from sphinx.ext.autodoc import Documenter, ModuleDocumenter, ModuleLevelDocumenter, ClassLevelDocumenter


from typing import cast
from sphinx.locale import _
from sphinx import addnodes
from sphinx.util.inspect import safe_getattr
from sphinx.util.docstrings import prepare_docstring
from sphinx.util.nodes import make_id
from sphinx.domains import ObjType
from sphinx.domains.python import PythonDomain, PyXRefRole, PyAttribute, PyClasslike
from sphinx.writers.html5 import HTML5Translator
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.util.fileutil import copy_asset
from docutils.parsers import rst
from docutils.parsers.rst import Directive, directives

from unittest import mock
import ast
import sys
import executing

VERBOSE = False
DOMAIN = 'con'
ALL = object()

construct_overrides = [
    construct.core.Struct,
    construct.core.Renamed,
    construct.core.Aligned,
    construct.core.Array,
    construct.core.FocusedSeq]

construct_inits = {}
for c in construct_overrides:
    construct_inits[c.__name__] = c.__init__


def Construct_mock__init__(self, *args, **kwargs):
    """
    This here is some introspection black magic to be injected in construct's
    init functions, only during documentation building.
    It allows construct class instances to remember a few things:
    1. The variable name they were instantiated as.
    2. The module they were defined as.
    """
    # Regular Init
    construct_inits[type(self).__name__](self, *args, **kwargs)
    # Additional init
    try:
        # Remember module it was declared in
        frame = sys._getframe(1)
        if isinstance(self, construct.core.FocusedSeq):
            frame = sys._getframe(2)
            self.subcons[1].__module__ = frame.f_globals['__name__']

        modname = frame.f_globals['__name__']
        if modname != 'construct.core':
            self.__module__ = modname
            for s in self.subcons:
                s.__module__ = self.__module__

        # Remember name it was declared as
        node = executing.Source.executing(frame).node
        while hasattr(node, 'parent') and not isinstance(node, ast.Assign):
            node = node.parent
            assert not isinstance(node, ast.Module)
        if isinstance(node.targets[0], ast.Name):
            self.name = node.targets[0].id
        elif isinstance(node.targets[0], ast.Attribute):
            self.name = node.targets[0].attr
    except Exception as e:
        node = None


def Subconstruct_mock__getattr__(self, name):
    """
    makes sure Subconstruct has __getattr__,
    otherwise introspection is impossible"""
    return getattr(self.subcon, name)


@contextmanager
def mocked_constructs():
    """A context to apply all the Construct mocks"""
    with ExitStack() as es:
        for x in construct_overrides:
            minit = mock.patch.object(x,
                                      '__init__', Construct_mock__init__)
            es.enter_context(minit)
        mgetattr = mock.patch.object(construct.core.Subconstruct,
                                     '__getattr__', Subconstruct_mock__getattr__,
                                     create=True)
        es.enter_context(mgetattr)
        yield es


def getdoc(obj, attrgetter=safe_getattr, allow_inherited=False):
    """method to extract a Struct's doc, replacing it's docstring."""
    return obj.docs


def hexify(val):
    binary = "{0:b}".format(int(val))
    return hex(len(binary))


def deconstruct(s, info=None):
    if not info:
        info = {
            'name': None,
            'varname': None,
            'value': None,
            'docs': None,
            'count': None,
            'const': None,
            'options': None}

    if isinstance(s, construct.core.FocusedSeq):
        return deconstruct(s.subcons[1], info)
    if isinstance(s, construct.core.Array):
        info['count'] = info.get('count') or []
        if isinstance(s.count, int):
            info['count'] = [s.count] + info['count']
        else:
            info['count'] = [''] + info['count']
    if isinstance(s, construct.core.Bytes):
        info['count'] = info.get('count') or []
        info['count'] = [s.length if isinstance(
            s.length, int) else ''] + info['count']

    if isinstance(s, construct.core.Enum):
        info['options'] = s.ksymapping
        info['options']['description'] = 'Represents an integer Enum'
    if isinstance(s, construct.core.FlagsEnum):
        info['options'] = {hexify(v): k for k, v in s.flags.items()}
        info['options']['description'] = 'Represents a flags Enum'
    if isinstance(s, construct.core.Const):
        info['const'] = s.value
    if isinstance(s, construct.core.Default):
        info['default'] = s.value

    if isinstance(s, construct.core.StringEncoded):
        info['varname'] = None
        return s, info
    elif isinstance(s, construct.core.Subconstruct):
        info['name'] = info['name'] or safe_getattr(s, 'name')
        info['docs'] = info['docs'] or safe_getattr(s, 'docs')
        return deconstruct(s.subcon, info)
    elif isinstance(s, construct.core.Struct):
        info['varname'] = safe_getattr(s, 'name', info['name'])
        return s, info
    else:
        info['varname'] = None
        return s, info


class MockedDocumenter(Documenter):
    def import_object(self, raiseerror: bool = False):  # valid
        with mocked_constructs() as a:
            return super().import_object()

    def get_object_members(self, want_all):
        return False, []


class ConstructDocumenter(MockedDocumenter):
    domain = DOMAIN
    member_order = 20

    def get_doc(self):
        # get the doctring
        docstring = getdoc(self.object, self.get_attr,
                           self.env.config.autodoc_inherit_docstrings)

        if docstring:
            tab_width = self.directive.state.document.settings.tab_width
            return [prepare_docstring(docstring, tab_width)]
        return []

    def generate(self, *args, **kwargs):
        # generate makes rst formated output
        super().generate(*args, **kwargs)

        if VERBOSE:
            print('-----result------')
            for l in self.directive.result:
                print(l)


class SubconDocumenter(ConstructDocumenter, ClassLevelDocumenter):
    objtype = 'subcon'
    priority = 11

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        # determine if member can be documented
        return isinstance(member, construct.core.Construct) and \
            isinstance(parent, StructDocumenter) and isattr

    def add_directive_header(self, sig):
        # add construct specific headers
        Documenter.add_directive_header(self, sig)

        sourcename = self.get_sourcename()
        s, infos = deconstruct(self.object)

        suffix = ''
        if 'const' in infos and infos['const'] is not None:
            self.add_line('   :const-value: ' +
                          str(infos['const']), sourcename)

        if 'default' in infos and infos['default'] is not None:
            self.add_line('   :default-value: ' +
                          str(infos['default']), sourcename)

        if 'count' in infos and infos['count']:
            suffix = ', [' + ']['.join([str(c) for c in infos['count']]) + ']'

        if 'options' in infos and infos['options']:
            options_string = json.dumps(
                infos['options'], separators=(',', ':'))
            self.add_line('   :field-options: ' + options_string, sourcename)

        if isinstance(s, construct.core.BitsInteger):
            self.add_line('   :field-type: bitint;' +
                          str(s.length) + suffix, sourcename)

        if isinstance(s, construct.core.Bytes):
            self.add_line('   :field-type: bytes' + suffix, sourcename)

        if s == construct.core.Flag:
            self.add_line('   :field-type: bool' + suffix, sourcename)

        if isinstance(s, construct.core.StringEncoded):
            if isinstance(s.subcon, construct.core.NullTerminated):
                self.add_line('   :string-type: ' +
                              'nullterminated' + suffix, sourcename)
            elif isinstance(s.subcon, construct.core.FixedSized):
                self.add_line('   :string-type: padded;' +
                              str(s.subcon.length) + suffix, sourcename)
        elif isinstance(s, construct.core.FormatField):
            self.add_line('   :field-type: ' +
                          s.fmtstr + suffix, sourcename)
        elif isinstance(s, construct.core.Struct) and 'varname' in infos:
            self.add_line('   :struct-type: ' +
                          (infos['varname'] or '') + suffix, sourcename)


class StructDocumenter(ConstructDocumenter, ModuleLevelDocumenter):
    objtype = 'struct'
    priority = 1

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        # determine if member can be documented
        return isinstance(member, construct.core.Construct) and \
            isinstance(parent, ModuleDocumenter) and isattr

    def filter_members(self, members, want_all):
        # filter out members fields not to be documented
        return [(membername, member, True) for membername, member in members
                if not member == construct.core.Pass and
                not isinstance(member, construct.core.Padded) and
                not isinstance(member, construct.core.Computed)]

    def get_object_members(self, want_all):
        # create a list of members out of self.object
        s, info = deconstruct(self.object)
        return (False, [(sinfo['name'], ssc) for ssc, sinfo in
                        [deconstruct(sc)[:4] for sc in s.subcons]])


class ModconDocumenter(MockedDocumenter, ModuleDocumenter):
    objtype = 'modcon'

    def add_directive_header(self, sig):
        return

    def filter_members(self, members, want_all):
        ret = []
        for mname, member in members:
            if self.object.__name__ == member.__module__:
                ret.append((mname, member, True))
            else:
                subcon = safe_getattr(member, 'subcon', None)
                if subcon and self.object.__name__ == subcon.__module__:
                    ret.append((mname, member.subcon, True))
        return ret

    def sort_members(self, documenters, order):
        documenters = super().sort_members(documenters, order)
        if order == 'bysource':
            documenters.reverse()
        return documenters

    def get_object_members(self, want_all):
        ret = []

        def isStruct(i):
            isS = isinstance(i, construct.core.Struct)
            isS = isS or isinstance(i, construct.core.Aligned)
            isS = isS or isinstance(i, construct.core.Renamed)
            isS = isS or isinstance(i, construct.core.Transformed)
            isS = isS or isinstance(i, construct.core.FocusedSeq)
            return isS

        for mname, member in inspect.getmembers(self.object, isStruct):
            ret.append((mname, safe_getattr(self.object, mname)))
        return False, ret


class desc_struct(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    pass


class desc_subcon(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    pass


class desc_stype(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    def astext(self):
        return '({})'.format(super().astext())


class desc_structref(desc_stype):
    pass


class desc_pytype(desc_stype):
    pass


class desc_value(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    pass


class desc_count(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    def astext(self):
        return '[{}]'.format(super().astext())


class desc_ctype(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    def astext(self):
        return '<parsed from {}>'.format(super().astext())


class desc_options(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    pass


class desc_option(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    pass


class desc_option_data(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    pass


class desc_option_data_desc(nodes.Part, nodes.Inline, nodes.FixedTextElement):
    pass


class StructHTML5Translator(HTML5Translator):

    def visit_desc_subcon(self, node):
        self.body.append(self.starttag(node, 'code', '',
                                       CLASS='sig-prename descclassname'))
        self.body.append('<span class="sig-paren">(</span>')

    def depart_desc_subcon(self, node):
        self.body.append('<span class="sig-paren">)</span></code>')

    def visit_type(self, node):
        self.body.append(self.starttag(node, 'code', '',
                                       CLASS='sig-prename descclassname'))

    def depart_type(self, node):
        self.body.append('</code>')

    visit_desc_structref = visit_type
    depart_desc_structref = depart_type
    visit_desc_pytype = visit_type
    depart_desc_pytype = depart_type

    def visit_desc_ctype(self, node):
        self.body.append(self.starttag(node, 'span', '',
                                       STYLE='color: gray'))
        self.body.append('&ltparsed from ')

    def depart_desc_ctype(self, node):
        self.body.append('&gt</span>')

    def visit_desc_count(self, node):
        pass

    def depart_desc_count(self, node):
        pass

    def visit_desc_options(self, node):
        self.body.append('<table class="con enum">')

    def depart_desc_options(self, node):
        self.body.append('</table>')

    def visit_desc_option(self, node):
        self.body.append('<tr>')

    def depart_desc_option(self, node):
        self.body.append('</tr>')

    def visit_desc_value(self, node):
        self.body.append('<span>, ')

    def depart_desc_value(self, node):
        self.body.append('</span>')

    def visit_desc_option_data(self, node):
        self.body.append('<td>')

    def depart_desc_option_data(self, node):
        self.body.append('</td>')

    def visit_desc_option_data_desc(self, node):
        self.body.append('<caption>')

    def depart_desc_option_data_desc(self, node):
        self.body.append('</caption>')


class StructStandaloneHTMLbuilder(StandaloneHTMLBuilder):
    @property
    def default_translator_class(self):
        return StructHTML5Translator


FF_TYPES = {
    # FormatField : (pytype, ctype)
    'e': ('float', '754 float'),
    'f': ('float', 'float'),
    'd': ('float', 'double'),
    'b': ('int', 'signed char'),
    'B': ('int', 'unsigned char'),
    'h': ('int', 'short'),
    'H': ('int', 'unsigned short'),
    'L': ('int', 'unsigned long'),
    'Q': ('int', 'integer'),
    'l': ('int', 'long'),
    'q': ('int', 'unsigned long long'),
}


def unformatCount(formatfieldstr):
    unformated = tuple(formatfieldstr.rsplit(', ', 1))
    if len(unformated) == 1:
        unformated += (None,)
    return unformated


def unformatFieldType(fieldtypestr):
    unformated = unformatCount(fieldtypestr)
    if unformated[0] == 'bytes':
        return ('bytes', None, unformated[-1])
    elif unformated[0] == 'bool':
        return ('bool', 'bool', unformated[-1])
    elif unformated[0].startswith('bitint'):
        length = unformated[0].rsplit(';', 1)[1]
        return ('int', 'a ' + str(length) + ' bit long integer', unformated[-1])
    else:
        return FF_TYPES[unformated[0][1]] + (unformated[-1],)


def unformatStringType(fieldstringstr):
    unformated = unformatCount(fieldstringstr)
    return ('string', unformated[0]) + (unformated[-1],)


def unformatFieldOptions(fieldoptionsstr):
    unformated = json.loads(fieldoptionsstr)
    desc = None
    if 'description' in unformated:
        desc = unformated['description']
        del unformated['description']
    return desc, unformated


class ConstructObjectDesc():
    domain = DOMAIN

    def add_target_and_index(self, name_cls, sig, signode):
        modname = self.options.get(
            'module', self.env.ref_context.get('py:module'))
        fullname = name_cls[0]
        node_id = make_id(self.env, self.state.document, '', fullname)

        # note target
        if fullname not in self.state.document.ids:
            signode['names'].append(fullname)
            signode['ids'].append(fullname)
            signode['first'] = (not self.names)
            self.state.document.note_explicit_target(signode)

            domain = cast(PythonDomain, self.env.get_domain(DOMAIN))
            domain.note_object(fullname, self.objtype, node_id)

        indextext = self.get_index_text(modname, name_cls)
        if indextext:
            self.indexnode['entries'].append(('single', indextext,
                                              fullname, '', None))


class Struct(ConstructObjectDesc, PyClasslike):
    option_spec = PyClasslike.option_spec.copy()


class Subcon(ConstructObjectDesc, PyAttribute):
    option_spec = PyAttribute.option_spec.copy()
    option_spec.update({
        'struct-type': rst.directives.unchanged,
        'string-type': rst.directives.unchanged,
        'field-type': rst.directives.unchanged,
        'field-options': rst.directives.unchanged,
        'const-value': rst.directives.unchanged,
        'default-value': rst.directives.unchanged
    })

    def handle_signature(self, sig, signode):

        fullname, prefix = super().handle_signature(sig, signode)
        subconnode = desc_subcon()

        struct_type = self.options.get('struct-type')
        if struct_type:
            stype, count = unformatCount(struct_type)
            refnode = addnodes.pending_xref('', refdomain=DOMAIN, refexplicit=False,
                                            reftype='struct', reftarget=stype)
            refnode += desc_structref(stype, stype)
            subconnode += refnode
            if count:
                subconnode += desc_count(count, count)
            signode += subconnode

        field_type = self.options.get('field-type')
        if field_type:
            pytype, ctype, count = unformatFieldType(field_type)

            refnode = addnodes.pending_xref('', refdomain='py', refexplicit=False,
                                            reftype='class', reftarget=pytype)
            refnode += desc_pytype(pytype, pytype)
            subconnode += refnode

            if count:
                subconnode += desc_count(count, count)

            signode += subconnode

            if ctype:
                signode += desc_ctype(ctype, ctype)

        string_type = self.options.get('string-type')
        if string_type:
            pytype, ctype, count = unformatStringType(string_type)

            refnode = addnodes.pending_xref('', refdomain='py', refexplicit=False,
                                            reftype='obj', reftarget=pytype)
            refnode += desc_pytype(pytype, pytype)
            subconnode += refnode
            if count:
                subconnode += desc_count(count, count)
            signode += subconnode
            if ctype == 'nullterminated':
                desc_string = 'a null terminated string'
            elif ctype.startswith('padded'):
                length = ctype.split(';')[1]
                desc_string = 'a ' + str(length) + ' bytes long padded string'
            else:
                desc_string = 'a string'
            signode += desc_ctype(ctype, desc_string)

        const = self.options.get('const-value')
        if const:
            subconnode += desc_value(const, 'constant=' + const)

        default = self.options.get('default-value')
        if default:
            subconnode += desc_value(default, 'default=' + default)

        return fullname, prefix

    def transform_content(self, contentnode):
        field_options = self.options.get('field-options')
        if field_options:
            d_options = desc_options()
            desc, content = unformatFieldOptions(field_options)
            d_options += desc_option_data_desc(desc, desc)
            for option in content.items():
                d_option = desc_option()
                d_option += desc_option_data(option[0], option[0])
                d_option += desc_option_data(option[1], option[1])
                d_options += d_option
            contentnode.insert(0, d_options)


class ConstructPythonDomain(PythonDomain):
    name = DOMAIN
    label = 'Construct'

    object_types = {'struct': ObjType(_('data'), 'data', 'obj'),
                    'subcon': ObjType(_('data'), 'data', 'obj')}
    directives = {'struct': Struct,
                  'subcon': Subcon}
    roles = {'struct': PyXRefRole(),
             'subcon': PyXRefRole()}


class init_directive(Directive):
    # used to import things in key areas before autodoc does
    has_content = True
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True

    def run(self):
        if len(self.arguments):
            with mocked_constructs():
                importlib.import_module(self.arguments[0])
        return []


def scb_static_path(app):
    app.config.html_static_path.append(
        str(Path(__file__).parent.joinpath("_static").absolute())
    )


def setup(app):
    app.connect("builder-inited", scb_static_path)
    app.add_builder(StructStandaloneHTMLbuilder, override=True)
    app.add_domain(ConstructPythonDomain)
    app.add_node(desc_structref)
    app.add_node(desc_pytype)
    app.add_node(desc_ctype)
    app.add_node(desc_count)
    app.add_node(desc_options)
    app.add_node(desc_option)
    app.add_node(desc_subcon)
    app.add_autodocumenter(ModconDocumenter)
    app.add_autodocumenter(StructDocumenter)
    app.add_autodocumenter(SubconDocumenter)
    app.add_directive('auto-construct-init', init_directive)

    app.add_css_file("sphinx-autodoc-construct.css")

    return {'version': sphinx.__display_version__,
            'parallel_read_safe': True,
            'parallel_write_safe': True}
