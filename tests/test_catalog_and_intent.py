import pandas as pd
import pytest

from stylematch.catalog import CatalogError, infer_category, load_catalog, normalize, normalize_gender, scrub_contacts
from stylematch.intent import parse_query
from stylematch.text import tokenize


def test_prototype_headers_are_accepted(tmp_path):
    path = tmp_path / "c.csv"
    pd.DataFrame(
        {
            "ProductID": [1, 2, 2],
            "ProductName": ["Brand Women Blue Kurta", "Brand Men Black Jeans", "dup"],
            "ProductBrand": ["Brand", "Brand", "Brand"],
            "Gender": ["Women", "Men", "Men"],
            "Price (INR)": [999, 1499, 1],
            "NumImages": [5, 5, 5],
            "Description": ["Blue kurta", "Black jeans", "x"],
            "PrimaryColor": [" Blue", None, "Red"],
        }
    ).to_csv(path, index=False)
    catalog = load_catalog(path)
    assert catalog["product_id"].tolist() == ["1", "2"]
    assert catalog.attrs["duplicates_removed"] == 1
    assert catalog["colour"].tolist() == ["blue", "unknown"]
    assert catalog["category"].tolist() == ["kurta", "jeans"]


def test_missing_column_and_bad_price_are_refused():
    with pytest.raises(CatalogError, match="price"):
        normalize(pd.DataFrame({"ProductID": [1], "ProductName": ["a"], "Gender": ["Men"]}))
    with pytest.raises(CatalogError, match="price must be"):
        normalize(pd.DataFrame({"ProductID": [1], "ProductName": ["a"], "Gender": ["Men"], "Price (INR)": ["free"]}))


def test_contact_details_are_removed_from_descriptions():
    text = scrub_contacts("Soft cotton top. For queries call customer care 1800 123 4567. Mail help@brand.example now")
    assert "1800" not in text and "@" not in text and "Soft cotton top" in text


@pytest.mark.parametrize(
    "name, category",
    [("X Men Navy T-Shirt", "tshirt"), ("X Women Sweatshirt", "sweatshirt"), ("X Formal Shirt", "shirt"), ("X Slim Fit Jeans", "jeans"), ("X Thing", "other")],
)
def test_category_inference(name, category):
    assert infer_category(name) == category


def test_gender_normalization():
    assert normalize_gender(" Unisex Kids ") == "unisex kids"
    assert normalize_gender("female") == "women"
    assert normalize_gender("???") == "unisex"


def test_tokenizer_folds_plurals():
    assert tokenize("Dresses and T-Shirts for parties") == ["dress", "tshirt", "party"]


def test_intent_parses_gender_budget_colour_category_and_occasion():
    intent = parse_query("Blue kurta for women under ₹2,000 for a wedding")
    assert intent.gender == "women"
    assert intent.max_price == 2000
    assert intent.colours == ["blue"]
    assert "kurta" in intent.categories
    assert intent.occasions == ["wedding"] and "saree" in intent.expansion


def test_intent_between_and_overrides():
    intent = parse_query("jeans between 1000 and 500", gender="men", max_price=900)
    assert (intent.min_price, intent.max_price, intent.gender) == (500, 900, "men")
    with pytest.raises(ValueError):
        parse_query("   ")
    with pytest.raises(ValueError):
        parse_query("shoes over 3000", max_price=100)


def test_kids_gender_maps_to_children_genders():
    assert parse_query("shoes for kids").allowed_genders() == ("boys", "girls", "unisex kids")
