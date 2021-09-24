import pandas as pd

data_path = "../data/metadata.csv"

df = pd.read_csv(data_path)
species_df = df[["Recording_ID", "Species"]]

# Selected species for project
selected_species_list = ["Fringilla coelebs", "Parus major", "Turdus merula", "Sylvia communis"]

selected_species_df = species_df[(species_df["Species"] == selected_species_list[0])
                                 | (species_df["Species"] == selected_species_list[1])
                                 | (species_df["Species"] == selected_species_list[2])
                                 | (species_df["Species"] == selected_species_list[3])]

id_list = selected_species_df["Recording_ID"].tolist()
species_list = selected_species_df["Species"].tolist()

species = []
for i in range(len(species_list)):
    id = id_list[i]
    specie = str(species_list[i]).replace(" ", "-")
    data = specie + "-" + str(id).strip()
    species.append([data])

pd.DataFrame(species, columns=["Species"]).to_csv("../data/species.csv", index=False)
print("done")
