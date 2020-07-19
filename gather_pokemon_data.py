import requests
import re
import pickle

URL_START = "https://bulbapedia.bulbagarden.net/wiki/"
URL_END = "_(Pok%C3%A9mon)"

entries_tag_pattern = r'<h\d><span class="mw-headline" id="Pok.C3.A9dex_entries(_\d)?">Pokédex entries</span></h\d>'

OUTSIDE_ENTRIES_STATE = "OUTSIDE_ENTRIES_STATE"
IN_ENTRIES_STATE = "IN_ENTRIES_STATE"
IN_TABLE_STATE = "IN_TABLE_STATE"


class PokemonInfo:
    def __init__(self, types):
        self.types = types  # set of types
        self.wiki_html = ""
        self.entries = list()  # list of pokedex entries


def main():
    pokemon_dict = get_pokemon_list()
    get_pokedex_entries(pokemon_dict)
    pickle.dump(pokemon_dict, open('data/pokemon_data.p', 'wb'))


def get_pokemon_list():
    pokemon_dict = {}
    with open("data/poke_list.tsv", "r") as file:
        lines = file.readlines()

    for line in lines:
        tokens = line.rstrip().split("\t")

        if len(tokens) >= 5:
            # Get pokemon with no Ndex number
            if "#" in tokens[1] or ():
                name = tokens[2]
                types = set(tokens[4:])

                # Add pokemon to collection
                if name not in pokemon_dict:
                    pokemon_dict[name] = PokemonInfo(types)
                else:
                    pokemon_dict[name].types = pokemon_dict[name].types.union(types)

    return pokemon_dict


def get_pokedex_entries(pokemon_dict):
    for pokemon in pokemon_dict:
        name = pokemon.replace(r"\s", "_")
        url = URL_START + name + URL_END

        html = requests.get(url)
        page_text = html.content.decode("utf-8")
        excavate_entries_from_html(page_text, pokemon_dict[pokemon])


def excavate_entries_from_html(page_text, pokemon_info):
    lines = page_text.split("\n")
    state = OUTSIDE_ENTRIES_STATE
    th_count = 0
    td_count = 0

    for line in lines:

        if state == OUTSIDE_ENTRIES_STATE:
            # Check for a relevant section
            match = re.search(entries_tag_pattern, line)
            if match:
                state = IN_ENTRIES_STATE
                continue

        if state == IN_ENTRIES_STATE or state == IN_TABLE_STATE:
            # Check for the end of the relevant section
            if line.startswith("<h"):
                state = OUTSIDE_ENTRIES_STATE
                continue

            if line.startswith("<tr"):
                state = IN_TABLE_STATE
                th_count = 0
                td_count = 0
                continue

        if state == IN_TABLE_STATE:
            # Look for entry lines by tracking number of table column headers (<th>) and table values (<tb>) within a table thing (<tr>)
            if line.startswith("<th"):
                th_count += 1
                continue

            elif line.startswith("<td"):
                td_count += 1

                if td_count > th_count:
                    td_count = 0

                elif td_count == th_count:
                    # Here's a pokedex entry!
                    get_entry_text(line, pokemon_info)


def get_entry_text(line, pokemon_info):
    match = re.search(">(.+)(<|$)", line)
    if match:
        text = match.group(1)
        if text not in pokemon_info.entries:
            pokemon_info.entries.append(text)
    # else:
    #     print("Non-matching td line: " + line)


if __name__ == '__main__':
    main()
