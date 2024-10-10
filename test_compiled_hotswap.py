from predict import Predictor


def test_compile_same_rank():
    """
    Runs a test with three different loras, all of which are the same rank. 
    """
    p = Predictor()
    p.setup(compile=True)
    p.predict(
        "a photo of a TOK dog",
        "1:1",
        None,
        None,
        1,
        28,
        3.5,
        1234,
        "png",
        80,
        "https://replicate.delivery/yhqm/hNeuharNetjpfpHVlUyPs8O8kygdciYIi8dNzj3K5bT8xllmA/trained_model.tar",
        1.1,
        True
    )
    p.predict(
        "a photo of a TOK dog",
        "1:1",
        None,
        None,
        1,
        28,
        3.5,
        1234,
        "png",
        80,
        "https://replicate.delivery/yhqm/0OmoRQJ60q6JOd6Fl9kIQ8P0W0vUQAMcu2s8uiSR8O5yJy0E/trained_model.tar",
        1.1,
        True
    )
    p.predict(
        "SHPS, A hawaiian beach in the style of SHPS",
        "1:1",
        None,
        None,
        1,
        28,
        3.5,
        1234,
        "png",
        80,
        "https://replicate.delivery/yhqm/3iisiVryWZr4C1oaVHZys6Q8gIaGquFEpeHjc68wioMpzgqJA/trained_model.tar",
        1.1,
        True
    )

def test_compile_different_rank():
    """
    Runs a lora of rank 16 and then a lora of rank 64
    """
    p = Predictor()
    p.setup(compile=True)
    p.predict(
        "a photo of a TOK dog",
        "1:1",
        None,
        None,
        1,
        28,
        3.5,
        1234,
        "png",
        80,
        "https://replicate.delivery/yhqm/hNeuharNetjpfpHVlUyPs8O8kygdciYIi8dNzj3K5bT8xllmA/trained_model.tar",
        1.1,
        True
    )
    p.predict(
        "a photo of ZIKI in the forest",
        "1:1",
        None,
        None,
        1,
        28,
        3.5,
        1234,
        "png",
        80,
        "https://replicate.delivery/yhqm/IosLs4j02TKeQSSiA0DxEsLKuf3fu0iJVd9Eelmqynoxf6naC/trained_model.tar",
        1.1,
        True
    )
