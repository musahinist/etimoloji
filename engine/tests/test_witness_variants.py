from engine.nlp.witness_variants import link_witness, split_forms, variant_cost


def test_split_forms_multi_form_heads():
    assert split_forms("uçmak, (uçmağ, uçmah)") == ["uçmak", "uçmağ", "uçmah"]
    assert split_forms("gez (I), (kez)") == ["gez", "kez"]
    assert split_forms("dilmaç (dilmeç (I))") == ["dilmaç", "dilmeç"]
    assert split_forms("pıçak") == ["pıçak"]
    assert split_forms("") == []


def test_cheap_dialect_substitutions():
    assert variant_cost("uçmağ", "uçmak") == 0.2  # söz sonu -ğ/-k
    assert variant_cost("pıçak", "bıçak") == 0.3
    assert variant_cost("depik", "tepik") == 0.3
    assert variant_cost("aldurmak", "aldırmak") == 0.4


def test_bound_rejects_distant_forms():
    assert variant_cost("kolluk", "kulluk", max_cost=0.9) == float("inf")
    assert variant_cost("elma", "armut") == float("inf")


def test_link_witness_uses_any_form_and_verb_stem():
    link = link_witness("gez (I), (kez)", "kez")
    assert link is not None and link.form == "kez" and link.cost == 0.0
    link = link_witness("gaçırmak", "kaçırmak")
    assert link is not None and link.cost == 0.3
    assert link_witness("armut", "elma") is None


def test_gloss_names_query_rejects_references_and_qualified_glosses():
    from engine.nlp.witness_variants import gloss_names_query

    assert gloss_names_query("Bıçak.", "bıçak")
    assert gloss_names_query("2. Yıl.", "yıl")
    assert gloss_names_query("Kaçırmak, kaybetmek", "kaçırmak")
    assert not gloss_names_query("Yakın", "yakınmak")
    assert not gloss_names_query("bk. gerçek", "gerçek")
    assert not gloss_names_query("1.[-> dé (II)]", "de")
    assert not gloss_names_query("O (Kuşu)", "o")


def test_dialect_names_query_needs_form_link_and_gloss():
    from engine.nlp.witness_variants import dialect_names_query

    assert dialect_names_query("pıçak", "Bıçak.", "bıçak") is not None
    assert dialect_names_query("garın", "Karın", "karın") is not None
    # anlam sorgu ama biçim bağlanmıyor
    assert dialect_names_query("sapı", "Bıçak.", "bıçak") is None
    # biçim bağlanıyor ama anlam başka (eşsesli)
    assert dialect_names_query("pir", "Cevizin yeşil kabuğu.", "bir") is None


def test_tarama_multi_form_head_is_split():
    from engine.fetchers.tdk_historical import _tarama_forms

    out = _tarama_forms("uçmak, (uçmağ, uçmah)")
    assert out["word"] == "uçmak" and out["variant_forms"] == ["uçmağ", "uçmah"]
    assert out["comparison"] == "uçmak" and out["attested_as"] == "uçmak, (uçmağ, uçmah)"
    # ana biçim sabit: iki varyant araması aynı tanığı verir
    assert _tarama_forms("gez (I), (kez)")["word"] == "gez"
    assert _tarama_forms("ağır (I)")["word"] == "ağır"
    assert _tarama_forms("pıçak") == {"word": "pıçak"}
