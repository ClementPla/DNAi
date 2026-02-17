from catppuccin.palette import PALETTE

palette = [c.hex for c in PALETTE.latte.colors]

TWO_COLORS = ["#99d1db", "#a6d189"]
THREE_COLORS = ["#99d1db", "#a6d189", "#e78284"]


class LabelsColors:
    IDU = "#a6d189"
    CLDU = "#ed8796"
    POST_TREATMENT = "#8aadf4"
    PRE_TREATMENT = "#949cbb"


class Grader:
    AI = "DNAI"
    HUMAN = "Human"
    OTHER = "FiberQ"
