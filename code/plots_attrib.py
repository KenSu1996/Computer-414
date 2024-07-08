import pandas as pd
import matplotlib.pyplot as plt


def generate_attribute_plots(csv_file_path, output_directory, plot_size=(10, 10)):
    data_frame = pd.read_csv(csv_file_path)

    attribute_list = ['race', 'race4', 'gender', 'age']
    for attribute in attribute_list:
        plt.figure(figsize=plot_size)
        plt.title(f"Distribution of {attribute.capitalize()}")
        data_frame[attribute].value_counts(normalize=True).sort_index(ascending=False).plot(kind='bar', rot=45,
                                                                                            legend=True)
        plt.savefig(f"{output_directory}/distribution_{attribute}.pdf", format='pdf')
        plt.close()



def run_plot_generation(source_csv, output_dir):
    generate_attribute_plots(source_csv, output_dir)


if __name__ == "__main__":
    src_csv = "./results/test_outputs.csv"
    output_dir = "./results"
    run_plot_generation(src_csv, output_dir)
