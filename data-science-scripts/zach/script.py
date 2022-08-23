is_gaussian_process_blueprints_enabled = True
nrows = 100
x_num = 10
ncols = 10
has_offset = False
get_is_weighted = False
rtype = 'Regression'
regression_types = ('Regression', 'MIN_INFLATED')

if all(
        [
            is_gaussian_process_blueprints_enabled is True,
            nrows <= 2000,
            x_num >= 1,
            ncols < 100,
            has_offset is False,
            get_is_weighted is False,
            rtype in regression_types,
        ]
    ):
    print("Adding the model to the repo")


if all(
        [
            is_gaussian_process_blueprints_enabled is True,
            nrows <= 2000,
            x_num >= 1,
            ncols < 100,
            has_offset is False,
            get_is_weighted is False,
            rtype in regression_types,
        ]
    ):
    print("Adding the model to the repo")
