"""Місце розмови = інша ФІЗИКА розмови.

Доти локації були декорацією: ядро про них не знало взагалі. Але «погомоніти в шинку» й «зібрати
віче на площі» — це не той самий процес у різних інтерʼєрах. У шинку немає старости, тому люди
кажуть те, чого при ньому не сказали б, і саме там народжуються чутки. У церкві розмова
віч-на-віч, і людина говорить, що думає НАСПРАВДІ, а не що казала вголос.

Кожен режим — це набір чисел і прапорців для ядра, а не опис настрою. Якщо режим не міняє нічого,
крім підпису, він не потрібен.
"""

from pydantic import BaseModel

DEFAULT_PLACE = "ploshcha"


class Mode(BaseModel):
    """Профіль розмови для одного місця."""

    place: str
    label: str
    # скільки людей приходить: у шинку купка, на вічі все село
    width: int = 6
    beats: tuple[int, int] = (12, 20)
    # температура зсувається, а не задається: базова лишається за умовою
    heat: float = 0.0
    interrupts: float = 1.0
    # староста зводить не всюди: у шинку модератора немає, і в цьому вся річ
    summary: bool = True
    doubt: bool = True
    # чи народжуються тут чутки (нема підстави — і ніхто не спинить)
    rumours: bool = True
    # підказка, що тут за розмова; іде в системне повідомлення мовця
    manner: str = ""


MODES: dict[str, Mode] = {
    "ploshcha": Mode(
        place="ploshcha", label="віче на Площі",
        manner="Це громадське віче: говорять по черзі, при старості, зважено."),
    "shynok": Mode(
        place="shynok", label="гомін у шинку", width=4, beats=(10, 16),
        heat=0.15, interrupts=1.6, summary=False, doubt=False,
        manner=("Це шинок, старости немає: перебивають, кажуть навпростець і те, чого при "
                "громаді не сказали б.")),
    "tserkva": Mode(
        place="tserkva", label="сповідь у церкві", width=2, beats=(6, 10),
        heat=-0.2, interrupts=0.2, summary=False, rumours=False,
        manner=("Це розмова віч-на-віч із попом. Кажи, що думаєш НАСПРАВДІ, а не те, що казав "
                "прилюдно. Тихо й без свідків.")),
    "kuznya": Mode(
        place="kuznya", label="діло в кузні", width=3, beats=(8, 12),
        heat=-0.1, interrupts=0.8, doubt=False, rumours=False,
        manner="Це не балачка, а діло: що робити руками, чим і коли."),
    "mlyn": Mode(
        place="mlyn", label="лічба в млині", width=3, beats=(8, 12),
        heat=-0.15, interrupts=0.6, doubt=False, rumours=False,
        manner="Тут рахують: скільки, почому, на скільки вистачить. Числа, а не здогади."),
    "dzvin": Mode(
        place="dzvin", label="тривога коло дзвону", width=5, beats=(6, 10),
        heat=0.25, interrupts=1.8, summary=True, doubt=False,
        manner="Ударили в дзвін: часу нема, кажи коротко й по суті."),
}


def mode_for(place: str | None) -> Mode:
    return MODES.get(str(place or DEFAULT_PLACE), MODES[DEFAULT_PLACE])
