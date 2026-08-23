

def test_a_female_name_never_lands_on_a_male_figure():
    """★ Малюнок ролі фіксований, імена вигадує модель — і вона плутала стать: «дід Свирид: як я
    ще дівкою була». Розсинхрон малюнка й підпису видно на екрані, тому вирішує код."""
    from ploshcha_sim.domain.people import fit_gender, name_gender

    assert name_gender("Одарка") == "ж"
    assert name_gender("дід Свирид") == "ч", "титул попереду не має вирішувати рід"
    assert name_gender("Микола") == "ч", "чоловіче імʼя на -а — виняток, а не жінка"

    assert fit_gender("did", "Одарка") == "дід Свирид", "жіноче імʼя на дідовій фігурі — заміна"
    assert fit_gender("mati", "Панас") == "Марія"
    assert fit_gender("koval", "Микола") == "Микола", "збіг статі — імʼя моделі лишається"
    assert fit_gender("hist", "будь-що") == "будь-що", "невідома роль — не судимо"
