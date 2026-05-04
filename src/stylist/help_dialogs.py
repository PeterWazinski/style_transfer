"""Help dialog functions for the Stylist application.

Each function creates and shows a modal dialog.  Pass the
:class:`~src.stylist.main_window.MainWindow` instance as *parent*.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def _show_link_dialog(parent: QWidget, title: str, html: str) -> None:
    """Show an informational dialog that supports clickable hyperlinks."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.setMinimumWidth(520)

    label = QLabel(html)
    label.setWordWrap(True)
    label.setOpenExternalLinks(True)
    label.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
    label.setContentsMargins(4, 4, 4, 4)

    scroll = QScrollArea()
    scroll.setWidget(label)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)

    ok_btn = QPushButton("OK")
    ok_btn.setFixedWidth(80)
    ok_btn.clicked.connect(dlg.accept)

    layout = QVBoxLayout(dlg)
    layout.addWidget(scroll)
    layout.addWidget(ok_btn, alignment=Qt.AlignmentFlag.AlignRight)

    dlg.exec()


def show_how_to_use(parent: QWidget) -> None:
    _show_link_dialog(
        parent,
        "How to Use",
        "<b>Basic workflow</b><br>"
        "<ol>"
        "<li><b>Open a photo</b> &mdash; click <b>\U0001f4c2</b> or use <i>File &rarr; Open Photo</i> "
        "(Ctrl+O).</li>"
        "<li><b>Select a style</b> &mdash; click a thumbnail in the Styles panel on the left. "
        "The style is pre-loaded so the first apply is fast.</li>"
        "<li><b>Apply</b> <b>\u25b6</b> &mdash; runs the neural style transfer on your photo. "
        "The result appears on the right side of the split view. Drag the divider to compare "
        "before and after.</li>"
        "<li><b>Save</b> <b>\U0001f4be</b> &mdash; saves the styled result to disk.</li>"
        "</ol>"
        "<br>"
        "<b>Strength slider (0 \u2013 300%)</b><br>"
        "Controls how strongly the style is blended into your photo.<br>"
        "&nbsp;&nbsp;\u2022 <b>0%</b> = original photo, no style applied.<br>"
        "&nbsp;&nbsp;\u2022 <b>100%</b> = full style as produced by the model (natural reference point).<br>"
        "&nbsp;&nbsp;\u2022 <b>&gt;100%</b> = style is extrapolated beyond the model output &mdash; "
        "colours and textures become more intense.<br>"
        "Release the slider to re-run the current step with the new value. "
        "Only the right-pane result is updated; the left pane stays unchanged for comparison.<br><br>"
        "<b>Re-Apply \u23e9 &mdash; chaining styles</b><br>"
        "After a first Apply you can select a <i>different</i> style and click Re-Apply. "
        "This uses the current styled result as the new input, painting a second style "
        "on top of the first. You can chain as many styles as you like.<br>"
        "The left pane automatically switches to show the previous result so you can "
        "compare each step.<br><br>"
        "<b>Undo \u21a9</b><br>"
        "Steps back through the last three Apply / Re-Apply operations. "
        "Strength slider adjustments are <i>not</i> counted as separate undo steps.<br><br>"
        "<b>Reset \u21ba</b><br>"
        "Reloads the original photo and discards all style filters, returning the canvas "
        "to its initial state.<br><br>"
        "<b>Style Chains</b><br>"
        "Copy the current style chain to the clipboard via <i>File &rarr; Style Chain to Clipboard</i>. "
        "The YAML can be saved as a <code>.yml</code> file and later re-applied via "
        "<i>File &rarr; Apply Style Chain\u2026</i>, or processed in batch via "
        "<code>BatchStyler.exe --apply-style-chain</code>.",
    )


def show_about_nst(parent: QWidget) -> None:
    _show_link_dialog(
        parent,
        "About Neural Style Transfer",
        "<b>How Neural Style Transfer works</b><br><br>"
        "Neural Style Transfer (NST) applies the visual texture of a <i>style image</i> "
        "(e.g. a painting) to your <i>content photo</i> while preserving its "
        "structure and shapes.<br><br>"
        "<b>Feed-forward network (Johnson et al., 2016)</b><br>"
        "Unlike the original iterative optimisation, this app uses a lightweight "
        "convolutional network trained specifically for each style. Once trained, "
        "a single forward pass transforms any photo in milliseconds — "
        "no per-image optimisation required.<br><br>"
        "<b>Tiled inference</b><br>"
        "To handle large photos without running out of GPU memory, the image is "
        "divided into overlapping tiles, each processed independently, "
        "then blended back together seamlessly.<br><br>"
        "<b>Strength slider</b><br>"
        "Blends the styled result with the original photo "
        "(0&nbsp;% = original, 100&nbsp;% = fully styled). "
        "Tile size and overlap can be tuned in <i>File &#8594; Settings</i>.<br><br>"
        "<b>References</b><br>"
        "&#8226; Gatys et al. (2015) &mdash; "
        "<a href='https://arxiv.org/pdf/1508.06576'>A Neural Algorithm of Artistic Style</a> "
        "&mdash; the original NST paper using iterative optimisation.<br>"
        "&#8226; Johnson et al. (2016) &mdash; "
        "<a href='https://arxiv.org/pdf/1603.08155'>Perceptual Losses for Real-Time Style Transfer and Super-Resolution</a> "
        "&mdash; the feed-forward network used in this app.<br>"
        "&#8226; Ulyanov et al. (2017) &mdash; "
        "<a href='https://arxiv.org/abs/1607.08022'>Instance Normalization: The Missing Ingredient for Fast Stylization</a> "
        "&mdash; Instance Normalization used in the feed-forward network in place of Batch Normalization.<br>"
        "&#8226; Kaggle notebook &mdash; "
        "<a href='https://www.kaggle.com/code/yashchoudhary/fast-neural-style-transfer'>Fast Neural Style Transfer</a> "
        "by Yash Choudhary.",
    )


def show_credits(parent: QWidget) -> None:
    _show_link_dialog(
        parent,
        "Credits",
        "<b>Peter's Picture Stylist</b><br><br>"
        "Pretrained ONNX models courtesy of:<br>"
        "&nbsp;&nbsp;<em>yakhyo/fast-neural-style-transfer</em> (MIT) &mdash; Fast Neural Style Transfer<br>"
        "&nbsp;&nbsp;<em>igreat/fast-style-transfer</em> (MIT) &mdash; Fast Neural Style Transfer<br><br>"
        "Additional pretrained models:<br>"
        "&nbsp;&nbsp;<b>CycleGAN</b> (BSD) &mdash; unpaired image-to-image translation; "
        "Monet, Van Gogh, C\u00e9zanne and Ukiyo-e styles.<br>"
        "&nbsp;&nbsp;Original paper: <em>Zhu et al., 2017</em> &mdash; "
        "<a href='https://junyanz.github.io/CycleGAN/'>junyanz.github.io/CycleGAN</a><br><br>"
        "&nbsp;&nbsp;<b>AnimeGAN v2</b> (MIT) &mdash; photo-to-anime conversion.<br>"
        "&nbsp;&nbsp;Original repo: "
        "<a href='https://github.com/TachibanaYoshino/AnimeGANv2'>github.com/TachibanaYoshino/AnimeGANv2</a><br><br>"
        "Training infrastructure:<br>"
        "&nbsp;&nbsp;<b>Kaggle</b> &mdash; free GPU compute (T4 x1) "
        "used to train new styles.<br><br>"
        "Built with Python, PySide6, and ONNX Runtime.<br><br>"
        "Special thanks:<br>"
        "&nbsp;&nbsp;<b>Claude Sonnet 4.6</b> (Anthropic) &mdash; coding assistance and advice.",
    )
