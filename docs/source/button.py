"""
sources:
 1. https://github.com/conda/conda-docs/blob/04613488d57688d77b0bc20618b9a4d5b56947ed/web/button.py
 2. https://stackoverflow.com/questions/25088113/make-a-css-button-a-link-in-sphinx
"""

from docutils import nodes
import jinja2
#from docutils.parsers.rst.directives import unchanged
from docutils.parsers.rst import Directive
from sphinx.application import Sphinx
from docutils.parsers.rst import roles
from docutils.parsers.rst.states import Inliner
from typing import List, Dict
import re

# from https://sourceforge.net/p/docutils/code/HEAD/tree/branches/plugins/docutils/parsers/rst/directives/__init__.py
def unchanged(argument):
    """
    Return the argument text, unchanged.
    (Directive option conversion function.)

    No argument implies empty string ("").
    """
    if argument is None:
        return u''
    else:
        return argument  # unchanged!

BUTTON_TEMPLATE = jinja2.Template("""
{% if link  %} 
    <a href="{{ link }}">    
        <button class="{{ class }}" {{ style |default('') }}>{{ text }}</button>
    </a>
{% else %}
    <button class="{{ class }}" {{ style |default('') }}>{{ text }}</button>
{% endif %}
""")


# placeholder node for document graph
class ButtonNode(nodes.Element):
    ...


class ButtonDirective(Directive):
    has_content = True
    required_arguments = 0

    """
    You have to set `option_spec` before you do the following
    .. xxx::
        :link:
        :class:
    """
    option_spec = {
        'link': unchanged,
        'class': unchanged,
        'style': unchanged,
    }

    # this will execute when your directive is encountered
    # it will insert a ButtonNode into the document that will
    # get visisted during the build phase
    def run(self):
        # env = self.state.document.settings.env
        # app: Sphinx = env.app
        # app.add_css_file('css/button/solid-blue-btn.css')  # or you can set it on the conf.py

        node = ButtonNode()
        node['text'] = ''.join(self.content.data)
        """
        .. button::
            :class: solid-blue-btn

            content.data[0]
            content.data[1]

        .. button:: content.data[0]
            :class: solid-blue-btn                
        """
        # style = "background-color:green; font-size: 30px;"

        node['class'] = self.options['class'].strip()
        node['link'] = self.options.get('link')
        if self.options.get('style'):
            style = 'style=' + self.options['style']  # style = "background-color:green; font-size: 30px;"
            node['style'] = style
        return [node]


# build phase visitor emits HTML to append to output
def html_visit_button_node(self, node: ButtonNode):
    # html = BUTTON_TEMPLATE.render(text=node['text'], link=node['link'])
    html = BUTTON_TEMPLATE.render({k: v for k, v in node.attributes.items()})
    self.body.append(''.join([_.strip() for _ in html.split('\n') if _.strip()]))
    raise nodes.SkipNode


def button_role(role, rawtext, text, line_no, in_liner: Inliner, options: Dict = None, content: List = None):
    # 'link="https://www.google.com/" style="background-color:green; font-size: 30px;"'
    regex = re.compile(f"(?:.*"
                       fr"(?P<link>link=\S*)|"
                       fr"(?P<style>style=\S*)"
                       f")",
                       )

    text, cls, *options = text.split('|')
    node = ButtonNode()  # nodes.paragraph
    node['class'] = cls.strip()
    node['text'] = text.rstrip()

    if options:
        options: str = options[0].strip()
        for attr in options.split(' '):
            m = regex.match(attr)
            if m:
                dict_option = m.groupdict()
                if dict_option.get('link'):
                    dict_option['link'] = dict_option['link'][5:]  # remove link=
                for key, value in dict_option.items():
                    if value:
                        node[key] = value.strip()
    return [node], []


def setup(app: Sphinx):

    """
    .. button::
        option1:
        option2:
    """
    app.add_node(ButtonNode,
                 # ↓ kwargs
                 html=(html_visit_button_node, lambda: None)
                 # other_2: Tuple[Callable, Callable],
                 # other_3: Tuple[visit_xxx, depart_xxx],
                 )
    app.add_directive('button', ButtonDirective)

    """
    :btn:``
    """
    roles.register_canonical_role('btn', button_role)
