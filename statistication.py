import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt



# Extracts the names from a file
def _find_names(path):
    KEY = "#NAME"
    with open(path) as f:
        for line in f:
            if line.startswith(KEY):
                return line.strip().split("\t")[1:]
    raise ValueError("No names header found in file")

# Extracts the variables from the file
def _find_vars(path):
    KEY = "#VAR"
    with open(path) as f:
        for line in f:
            if line.startswith(KEY):
                return line.strip().split("\t")[1:]
    raise ValueError("No variables header found in file")

# Extracts the units header from a file
def _find_units(path):
    KEY = "#UNIT"
    with open(path) as f:
        for line in f:
            if line.startswith(KEY):
                return line.strip().split("\t")[1:]
    raise ValueError("No units header found in file")

# Forces an input of a certain type
def f_input(__prompt: object = "", variable_type=str):
    while True:
        try: return variable_type(input(__prompt))
        except ValueError: 
            print("Invalid input")
            continue

# Clears the console
def clear_screen(): os.system("cls" if os.name == "nt" else "clear")



class DataColumn:
    def __init__(self, name, var, unit, data) -> None:
        self.name = name
        self.var = var
        self.unit = unit
        self.data = data
    
    def __repr__(self) -> str:
        return f"DataColumn({self.name}, {self.var}, {self.unit})"



class RegressionFunctions:
    def squareroot(x,a,b,c):
        return a*np.sqrt(x+c) + b

    def linear(x,a,b):
        return a*x + b
    
    def quadratic(x,a,b,c):
        return a*x**2 + b*x + c
    
    def cubic(x,a,b,c,d):
        return a*x**3 + b*x**2 + c*x + d
    
    def exponential(x,a,b,c):
        return a*np.exp(b*x) + c
    
    def logarithmic(x,a,b,c):
        return a*np.log(b*x) + c
    
    def power(x,a,b,c):
        return a*x**b + c



class ColumnFunctions:
    def add_constant(column: DataColumn, constant: float) -> DataColumn:
        return DataColumn(column.name, column.var, column.unit, column.data + constant)

    def subtract_constant(column: DataColumn, constant: float) -> DataColumn:
        return DataColumn(column.name, column.var, column.unit, column.data - constant)
    
    def multiply_constant(column: DataColumn, constant: float) -> DataColumn:
        return DataColumn(column.name, column.var, column.unit, column.data * constant)
    
    def divide_constant(column: DataColumn, constant: float) -> DataColumn:
        return DataColumn(column.name, column.var, column.unit, column.data / constant)
    
    def square(column: DataColumn) -> DataColumn:
        return DataColumn(f"{column.name}^2", f"({column.var})^2", f"{column.unit}^2", column.data**2)

    def square_root(column: DataColumn) -> DataColumn:
        return DataColumn(f"sqrt({column.name})", f"sqrt({column.var})", f"sqrt({column.unit})", np.sqrt(column.data))
    
    def log(column: DataColumn) -> DataColumn:
        return DataColumn(f"log({column.name})", f"log({column.var})", f"log({column.unit})", np.log(column.data))
    
    def ln(column: DataColumn) -> DataColumn:
        return DataColumn(f"ln({column.name})", f"ln({column.var})", f"ln({column.unit})", np.log(column.data))
    
    def exp(column: DataColumn) -> DataColumn:
        return DataColumn(f"exp({column.name})", f"exp({column.var})", f"exp({column.unit})", np.exp(column.data))
    
    def sin(column: DataColumn) -> DataColumn:
        return DataColumn(f"sin({column.name})", f"sin({column.var})", f"sin({column.unit})", np.sin(column.data))
    
    def cos(column: DataColumn) -> DataColumn:
        return DataColumn(f"cos({column.name})", f"cos({column.var})", f"cos({column.unit})", np.cos(column.data))
    
    def tan(column: DataColumn) -> DataColumn:
        return DataColumn(f"tan({column.name})", f"tan({column.var})", f"tan({column.unit})", np.tan(column.data))
    
    def arcsin(column: DataColumn) -> DataColumn:
        return DataColumn(f"arcsin({column.name})", f"arcsin({column.var})", f"arcsin({column.unit})", np.arcsin(column.data))
    
    def arccos(column: DataColumn) -> DataColumn:
        return DataColumn(f"arccos({column.name})", f"arccos({column.var})", f"arccos({column.unit})", np.arccos(column.data))
    
    def arctan(column: DataColumn) -> DataColumn:
        return DataColumn(f"arctan({column.name})", f"arctan({column.var})", f"arctan({column.unit})", np.arctan(column.data))



class Application:
    def __init__(self) -> None:
        self.stored_data = []
        self.last_console_message = ""
        self.AVAILABLE_FUNCTIONS = [function for function in ColumnFunctions.__dict__.keys() if not function.startswith("_")]
        self.AVAILABLE_REGESSIONS = [function for function in RegressionFunctions.__dict__.keys() if not function.startswith("_")]
    
    # Application columns
    def _create_column(self, name: str, var: str, unit: str, data) -> DataColumn:
        return DataColumn(name, var, unit, data)
    
    def _add_column(self, name: str, var: str, unit: str, data) -> None:
        self.stored_data.append(self._create_column(name, var, unit, data))
    
    def _remove_column(self, index: int) -> None:
        self.stored_data.pop(index)
    
    def _modify_column(self, index: int, function, overwrite: bool) -> None:
        modified_column = function(self.stored_data[index])
        if overwrite:
            self.stored_data[index] = modified_column
        else:
            self.stored_data.append(modified_column)
    
    def _request_column(self, index: int) -> DataColumn | None:
        if index == -1:
            return None
        elif index >= len(self.stored_data) or index < 0:
            raise IndexError("Index out of range")
        else:
            return self.stored_data[index]


    # Data IO
    def _import_data(self, path) -> None:

        # Validation check
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist")

        # Load data
        try:
            names = _find_names(path)
            variables = _find_vars(path)
            units = _find_units(path)
            datum = np.loadtxt(path, unpack=True)
        except ValueError:
            raise ValueError("File does not contain the correct keys")

        # Validation check
        all_same_dimensions = len(set([len(names), len(variables), len(units), len(datum)])) == 1
        if not all_same_dimensions:
            raise ValueError("Not every column contains the same amount of data")
    
        # Save data to application
        for name, var, unit, data in zip(names, variables, units, datum):
            self._add_column(name, var, unit, data)
    
    def _export_data(self, path) -> None:
        # TODO test with different size data columns
        names_list = []
        variables_list = []
        units_list = []
        data_list = []
        for column in self.stored_data:
            names_list.append(column.name)
            variables_list.append(column.var)
            units_list.append(column.unit)
            data_list.append(column.data)
        
        names = "\t".join(names_list)
        variables = "\t".join(variables_list)
        units = "\t".join(units_list)
        data = "\n".join(["\t".join([str(x) for x in row]) for row in np.array(data_list).T])

        with open(path, "w") as f:
            f.write(f"#NAME\t{names}\n")
            f.write(f"#VAR\t{variables}\n")
            f.write(f"#UNIT\t{units}\n")
            f.write(data)


    # Listing
    def list_available_columns(self) -> None:
        print("Available columns:")
        for i, column in enumerate(self.stored_data):
            print(f"{i}: {column.name} ({column.var}) [{column.unit}]")

    def list_available_functions(self) -> None:
        print("Available functions:")
        for i, function in enumerate(self.AVAILABLE_FUNCTIONS):
            print(f"{i}: {function}")
    
    def list_available_regressions(self) -> None:
        print("Available regressions:")
        for i, function in enumerate(self.AVAILABLE_REGESSIONS):
            print(f"{i}: {function}")


    # Querying
    def user_query_column(self, txt="Enter column index", display_list=False) -> DataColumn | None:
        if display_list: self.list_available_columns()
        while True:
            index = f_input(txt, int)
            try:
                return self._request_column(index)
            except IndexError:
                print("Index out of range")
                continue

    def user_query_function(self):
        self.list_available_functions()
        index = f_input("Enter function index: ", int)
        function = self.AVAILABLE_FUNCTIONS[index]
        return getattr(ColumnFunctions, function)
    
    def user_query_regression(self):
        self.list_available_regressions()
        index = f_input("Enter regression index: ", int)
        regression = self.AVAILABLE_REGESSIONS[index]
        return getattr(RegressionFunctions, regression)

    def user_query_scatterplot(self) -> tuple[DataColumn, DataColumn, DataColumn]:

        # List columns
        self.list_available_columns()

        # Query columns
        dependent_column = self.user_query_column("Enter dependent column index: ")
        independent_column = self.user_query_column("Enter independent column index: ")
        query_error = self.user_query_column("Enter independent error column (-1 for manual): ")
        
        # Manual error
        if query_error is None:
            error_name = f"Error of {independent_column.name}"
            error_var = f"d{independent_column.var}"
            error_unit = f"{independent_column.unit}"
            error_value = f_input("Enter the error in the independent variable: ", float)
            error_data = np.full(len(independent_column.data), error_value)
            independent_error_column = DataColumn(error_name, error_var, error_unit, error_data)
            # self._add_column(error_name, error_var, error_unit, error_data)
        else:
            independent_error_column = query_error
        
        # Return columns
        return dependent_column, independent_column, independent_error_column
    

    # Console IO
    def user_add_column(self) -> None:

        # Query column
        name = input("Enter column name: ")
        var = input("Enter column variable: ")
        unit = input("Enter column unit: ")
        length = f_input("Enter column length: ", int)
        value = f_input("Enter column value: ", float)
        
        # Add column
        data = np.full(length, value)
        self._add_column(name, var, unit, data)
        self.last_console_message = "Column added successfully!"

    def user_remove_column(self) -> None:

        # Validation check
        if len(self.stored_data) == 0:
            self.last_console_message = "Stored data is empty!"
            return

        # Remove column
        self.list_available_columns()
        index = f_input("Enter column index to remove: ", int)
        self._remove_column(index)
        self.last_console_message = "Column removed successfully!"

    def user_modify_column(self) -> None:

        # Validation check
        if len(self.stored_data) == 0:
            self.last_console_message = "Stored data is empty!"
            return
        
        # Modify column
        self.list_available_columns()
        index = f_input("Enter column index to modify: ", int)
        function = self.user_query_function()
        overwrite = input("Overwrite column? (y/n) ")
        overwrite = bool(overwrite == "y")
        self._modify_column(index, function, overwrite)
        self.last_console_message = "Column modified successfully!"

    def user_import_data(self) -> None:
        print("Drag and Drop the file you want to analyze into the terminal window and press enter")
        path = input("Enter file path: ")
        self._import_data(path)
        self.last_console_message = "Data imported successfully!"

    def user_export_data(self) -> None:

        # Validation check
        if len(self.stored_data) == 0:
            self.last_console_message = "Stored data is empty!"
            return
        
        # Export data
        print("Ender the file path you want to export to")
        path = input("Enter file path: ")
        self._export_data(path)
        self.last_console_message = "Data exported successfully!"

    def user_plot_regression(self) -> None:

        # Validation check
        if len(self.stored_data) == 0:
            self.last_console_message = "Stored data is empty!"
            return
        
        # Query data
        dependent_column, independent_column, independent_error_column = self.user_query_scatterplot()
        regression = self.user_query_regression()

        # Plot text objects
        x_label = f"{dependent_column.name}\n{dependent_column.var}({dependent_column.unit})"
        y_label = f"{independent_column.var}({independent_column.unit})\n{independent_column.name}"
        title = f"{independent_column.name} vs. {dependent_column.name}"

        # Create plot
        fig = plt
        fig.errorbar(dependent_column.data, independent_column.data, yerr=independent_error_column.data, capsize=5, marker='o', linestyle='None')
        fig.xlabel(x_label)
        fig.ylabel(y_label)
        fig.title(title)

        # Regression
        fig_params, covariance = opt.curve_fit(regression, dependent_column.data, independent_column.data, sigma = independent_error_column.data)
        fig.plot(dependent_column.data, regression(dependent_column.data, *fig_params), label = 'fit')
        
        # Calculate statistics
        fig_param_errors = np.sqrt(np.diag(covariance))
        chi_squared = np.sum(((independent_column.data - regression(dependent_column.data, *fig_params)) / independent_error_column.data)**2)
        degrees_of_freedom = len(dependent_column.data) - len(fig_params)
        reduced_chi_squared = chi_squared / degrees_of_freedom

        # Print statistics
        [print(f"{fig_params[i]:.3f} +/- {fig_param_errors[i]:.3f}") for i in range(len(fig_params))]
        print(f"chi-squared = {chi_squared:.3f}")
        print(f"degrees of freedom = {degrees_of_freedom}")
        print(f"reduced chi-squared = {reduced_chi_squared:.3f}")

        # Calculate r^2
        residuals = independent_column.data - regression(dependent_column.data, *fig_params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((independent_column.data - np.mean(independent_column.data))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"r^2 = {r_squared:.3f}")

        # Show figure
        fig.show()
        self.last_console_message = "Regression plotted successfully!"

    def user_plot_scatter(self) -> None:

        # Validation check
        if len(self.stored_data) == 0:
            self.last_console_message = "Stored data is empty!"
            return
        
        # Query data
        dependent_column, independent_column, independent_error_column = self.user_query_scatterplot()

        # Plot text objects
        x_label = f"{dependent_column.name}\n{dependent_column.var}({dependent_column.unit})"
        y_label = f"{independent_column.var}({independent_column.unit})\n{independent_column.name}"
        title = f"{independent_column.name} vs. {dependent_column.name}"

        # Create plot
        fig = plt
        fig.errorbar(dependent_column.data, independent_column.data, yerr=independent_error_column.data, capsize=5, marker='o', linestyle='None')
        fig.xlabel(x_label)
        fig.ylabel(y_label)
        fig.title(title)

        # Show figure
        fig.show()
        self.last_console_message = "Scatter plot plotted successfully!"

    def user_list_column(self) -> None:

        # Validation check
        if len(self.stored_data) == 0:
            self.last_console_message = "Stored data is empty!"
            return

        # Query column
        self.list_available_columns()
        index = f_input("Enter column index to list: ", int)
        column = self._request_column(index)
        
        # Print column
        clear_screen()
        print(f"Name: {column.name}")
        print(f"Variable: {column.var}")
        print(f"Unit: {column.unit}")
        print(f"Data: {column.data}")
        input("Press enter to continue...")



# Main
def main():
    app = Application()
    while True:

        # Clear console
        clear_screen()
        print(f"Console message: {app.last_console_message}\n")
        app.last_console_message = ""

        # Print available commands
        print("Available commands:")
        print("1: Import Data")
        print("2: Export Data")
        print("3: Add Column")
        print("4: Remove Column")
        print("5: Modify Column")
        print("6: List Column")
        print("7: Plot Scatter")
        print("8: Plot Regression")
        print("9: Quit")

        # Get user input
        option = f_input("> ", int)

        # Import Data
        clear_screen()
        if option == 1:
            app.user_import_data()
        elif option == 2:
            app.user_export_data()
        elif option == 3:
            app.user_add_column()
        elif option == 4:
            app.user_remove_column()
        elif option == 5:
            app.user_modify_column()
        elif option == 6:
            app.user_list_column()
        elif option == 7:
            app.user_plot_scatter()
        elif option == 8:
            app.user_plot_regression()
        elif option == 9:
            break
        else:
            print("Invalid option")


# Main guard
if __name__ == "__main__":
    main()