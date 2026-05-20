from collections.abc import Sequence
from typing import List

import matplotlib
import matplotlib.cm
import matplotlib.colors as mcolors
import numpy as np


class MidpointNormalizer(mcolors.Normalize):
    """
    Custom normalization class to shift the midpoint of a colormap.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=1, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Normalize the data based on the midpoint
        result, is_scalar = self.process_value(value)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
        rescaled_value = np.ma.masked_array(
            np.interp(value, [vmin, midpoint, vmax], [0, 0.5, 1])
        )
        return np.ma.filled(rescaled_value, 0) if is_scalar else rescaled_value


def colorize_weighted_text(
    words: Sequence,
    weights: Sequence,
    aspan_flags,
    dspan_flags,
    text_on_hover=None,
    cmap=matplotlib.colormaps.get_cmap("Greens"),
    color_normalizing_factor=1.0,
) -> str:
    """
    Generates HTML for the colorized text based on weights.
    """
    template = '<span title="{}" class="barcode" style="color: {}; background-color: {}; font-size:1.5em">{}</span>\n'
    colored_string = ""
    assert len(words) == len(weights), (
        "words and color array should be the same length!"
    )
    if text_on_hover is None:
        text_on_hover = weights
    for word, weight, aspan_flag, dspan_flag, weight_on_hover in zip(
        words, weights, aspan_flags, dspan_flags, text_on_hover
    ):
        background_color = matplotlib.colors.rgb2hex(
            cmap(weight / color_normalizing_factor)[:3]
        )
        if aspan_flag in ["True", "TRUE", True]:
            word_color = "red"
        elif dspan_flag in ["True", "TRUE", True]:
            word_color = "purple"
        elif weight_on_hover / color_normalizing_factor > 0.7:
            word_color = "white"
        else:
            word_color = "black"
        colored_string += template.format(
            round(weight_on_hover, 2),
            word_color,
            background_color,
            "&nbsp" + word + "&nbsp",
        )
    return colored_string


def generate_html_for_texts(
    titles,
    texts,
    weights_list,
    output_file_name="colorized_texts.html",
    color_normalizing_factor=1.0,
    cmap=matplotlib.colormaps.get_cmap("Greens"),
    additional_note: str = "",
    aspan_flags_list: List[List[bool]] | None = None,
):
    """
    Generates an HTML file that includes the colorized text for each input text and its corresponding weights.
    """
    html_content = "<meta charset='utf-8'>\n"
    if aspan_flags_list is None:
        aspan_flags_list = [None] * len(texts)
    for title, text, weights, aspan_flags in zip(
        titles, texts, weights_list, aspan_flags_list
    ):
        words = text.split()
        if aspan_flags is None:
            aspan_flags = [False] * len(words)
        word_num = len(words)
        dspan_flags = [False] * word_num  # Example flags

        html_content += f"<h2>{title}:</h2>\n"
        colorized_text = colorize_weighted_text(
            words,
            weights,
            aspan_flags,
            dspan_flags,
            cmap=cmap,
            color_normalizing_factor=color_normalizing_factor,
        )
        html_content += colorized_text + "<br><br>\n"

    html_content += f"<p>{additional_note}</p>"
    # Save the final HTML content
    with open(output_file_name, "w") as f:
        f.write(html_content)


if __name__ == "__main__":
    # Example usage
    texts = [
        "Many of us know we don't get enough sleep, but imagine if there was a simple solution.",
        "In a speech at the British Science Festival, Dr. Paul Kelley from Oxford University said schools should stagger their starting times.",
    ]
    weights_list = [
        [x / len(range(len(texts[0].split()))) for x in range(len(texts[0].split()))],
        [x / len(range(len(texts[1].split()))) for x in range(len(texts[1].split()))],
    ]
    titles = ["Sleep Problem", "British Science Festival Speech"]

    generate_html_for_texts(titles, texts, weights_list)
