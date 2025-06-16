# importing the necessary libraries and given files
import pandas as pd
import numpy as np
from sim_parameters import *
from helper import *


class CovidSimulation:
    '''
        # this class is very similar to assignment 2
        # this class will simulate the covid simulation
    '''

    def __init__(self, transition_probabilities, holding_times):

        # getting attributes by arguments of the constructor
        self.transtion_probabilities = transition_probabilities
        self.holding_times = holding_times

        # check if the transtion_probabilities are sum upto 1
        # if not, throw a RuntimeError

        for key in transition_probabilities:

            # get the inner dictionary related to the current key
            inner_dict = transition_probabilities[key]

            # total of probabilities are 1
            total = 0

            # calculate the total
            for inner_key in inner_dict:
                total += inner_dict[inner_key]

            # check if the toatal is not 1
            if total != 1:

                # throw a RuntimeError
                raise RuntimeError

        # set initial state to sunny
        self.current_state_attr = list(self.transtion_probabilities.keys())[0]

       # set the elapsed time to 0 for the current state
        self.elapsed_time_for_curr_state = 0

    # getting the states in the system
    def get_states(self):

        return list(self.transtion_probabilities.keys())

    # getting the current state
    def current_state(self):

        return self.current_state_attr

    # changing the current state to a new state
    def next_state(self):

        # if the remaining hours are not 0 , then the current state should stay for a while
        if self.current_state_remaining_hours() != 0:

            # but we need to reduce the hours by 1
            self.elapsed_time_for_curr_state -= 1
            return

        # otherwise update the next state

        # getting the probability related to the current state
        probs_related_to_curr_state = self.transtion_probabilities[self.current_state_attr]

        # key(weather list)
        weather_states = list(probs_related_to_curr_state.keys())

        # value(probability list)
        probabilities = list(probs_related_to_curr_state.values())

        # generate a random probability
        next_state_attr = np.random.choice(weather_states, p=probabilities)

        # set the new state
        self.set_state(next_state_attr)

        # update the remaining hours

        # get the index related to the current state
        index = self.get_states().index(self.current_state_attr)

        # get the holding time related to the current index
        holding_time = self.holding_times[self.get_states()[index]]

        # if the holding time is 0 , then update the elapsed time to the new holding time for the new state
        if holding_time == 0:
            self.elapsed_time_for_curr_state = 0
        # otherwise reduce one from the current elapsed time
        else:
            self.elapsed_time_for_curr_state = holding_time - 1

        return holding_time

    # setting the current state to a new state
    # can be used when implementing next state

    def set_state(self, new_state):
        self.current_state_attr = new_state

    def current_state_remaining_hours(self):

        # return remaning hours
        return self.elapsed_time_for_curr_state

    def iterable(self):

        while True:
            # yield current state
            yield self.current_state()
            # apply next_state()
            yield self.next_state()

    def simulate(self, days):

        # lists for storing states and next states
        states = []
        next_states = []

        # get the iteration
        iter = self.iterable()

        # run the simulation number of times
        for _ in range(days):

            # get the nextstate
            state = next(iter)

            # also call this after the next state
            nx_state = next(iter)

            next_states.append(nx_state)
            # append this next state to the health list
            states.append(state)

        # return the two lists
        return states, next_states

# helper function to create the nested dictionary


def nested_dictionary(my_dict):

    # initially all the states are 0
    output_dict = {'H': 0, 'I': 0, 'S': 0, 'D': 0, 'M': 0}

    # get the output dictionaries key list as the states
    output_dict_states = list(output_dict.keys())

    # get the my dictionary keys as a list
    my_dict_states = list(my_dict.keys())

    # go thorugh each key combination
    for i in range(len(my_dict_states)):
        for j in range(len(output_dict_states)):

            # if the keys are similar
            if output_dict_states[j] == my_dict_states[i]:

                # set the output dictionary to the given dictioanry entry
                output_dict[output_dict_states[j]
                            ] = my_dict[output_dict_states[j]]

                # since the correct one is found, break the inner loop
                break

    # return the updated dictionary
    return output_dict


def create_simulated_csv(people_count, day_count, given_country_info, start_date_df, transition_probability, age_probabilities, hold_times, hold_ages):
    '''
        # helper function to create the time series csv
    '''

    # for storing the date in the simulation
    output_df = pd.DataFrame(index=range(
        people_count*day_count), columns=range(7))

    # set the name of the columns
    output_df = output_df.rename(
        columns={0: 'id', 1: 'age_group_name', 2: 'country', 3: 'date', 4: 'state', 5: 'staying_days', 6: 'prev_state'})

    # for storing the number of objects and the current row
    object_counter, curr_row = 0, 0

    # loop through every country in the given list
    for curr_country in range(len(given_country_info)):

        # search for every age group in the list
        for curr_age_group in range(3, 8):

            # look for every entry
            for entry in range(given_country_info.iat[curr_country, curr_age_group]):

                # get the current date dataframe for every entry
                curr_date_df = start_date_df

                # get the related transtion probability
                curr_trainsition_probability = transition_probability[
                    age_probabilities[curr_age_group-3]]

                # get the related holding times
                curr_holding_times = hold_times[hold_ages[curr_age_group-3]]

                # then create a covid simulation object for it
                simulation_object_for_entry = CovidSimulation(
                    curr_trainsition_probability, curr_holding_times)

                # then apply the simulation for it and get the results
                simulate, summary = simulation_object_for_entry.simulate(
                    day_count)

                # set the initial state and inster intial holding time
                # initially all the people are healthy people
                # therefore set it as to "H"
                summary.insert(
                    0, hold_times[hold_ages[curr_age_group-3]]['H'])

                # count the number of days
                day_counter = 1

                # get each of the date
                for curr_date in range(len(simulate)):

                    # start to fill the output data frame

                    # set the entry number to indexing
                    output_df.iat[curr_row, 0] = object_counter

                    # set the correct age
                    output_df.iat[curr_row,
                                  1] = given_country_info.columns[curr_age_group]

                    # get the current country
                    output_df.iat[curr_row,
                                  2] = given_country_info.iat[curr_country, 0]

                    # set the current date
                    output_df.iat[curr_row, 3] = str(curr_date_df[0])[:10]

                    # set the time delta to be 1 day
                    curr_date_df = curr_date_df + pd.Timedelta("1 day")

                    # set the holding value
                    output_df.iat[curr_row, 4] = str(simulate[curr_date])

                    # check for the value in the summary for the current date
                    if summary[curr_date] == None:

                        # if it is not contain anything, update the day counter by 1
                        day_counter += 1

                    # otherwise
                    else:

                        # set the day counter to be 1
                        day_counter = 1

                    # update the day counter in the data frame
                    output_df.iat[curr_row, 5] = day_counter

                    # initial state is always healthy
                    # if the current date is the first day, the state is obviously healthy
                    if curr_date == 0:
                        output_df.iat[curr_row, 6] = "H"

                    # if the day counter is 1, then get the next state from the simulated list
                    elif day_counter == 1:
                        output_df.iat[curr_row, 6] = str(
                            simulate[curr_date-1])
                        prev_state = str(simulate[curr_date-1])

                    # otherwise set to the previous state
                    else:
                        output_df.iat[curr_row, 6] = prev_state

                    # increment the row value for the dataframe updating purposes
                    curr_row += 1

                # also increment the object counter
                object_counter += 1

    # save the created data frame to a csv file

    output_df.to_csv("a3-covid-simulated-timeseries.csv", index=False)

    return output_df


def create_summary_to_csv(countries, day_count, start_date_df, given_country_info, simulation):
    '''
        # helper function to save the summary csv
    '''

    # create the empty dataframe to store the output
    output_dframe = pd.DataFrame(index=range(
        len(countries)*day_count), columns=range(7))

    # add the columns to it
    output_dframe = output_dframe.rename(
        columns={0: 'date', 1: 'country', 2: 'D', 3: 'H', 4: 'I', 5: 'M', 6: 'S'})

    # variables to store the row number and current dates
    curr_row = 0
    curr_date_df = start_date_df

    # for each country
    for _ in range(day_count):

        # for each age group
        for curr_age_group in range(len(given_country_info)):

            # create a new dataframe with related given date
            given_date_info = simulation[simulation['country']
                                         == given_country_info.iat[curr_age_group, 0]]

            # filter out the date for a given date
            given_date_info = given_date_info[given_date_info['date'] == str(curr_date_df[0])[
                : 10]]

            # add the date
            output_dframe.iat[curr_row, 0] = str(curr_date_df[0])[: 10]

            # add the country
            output_dframe.iat[curr_row,
                              1] = given_country_info.iat[curr_age_group, 0]

            # get the values time series
            values_series = given_date_info['state'].value_counts()

            # get the index list of the series after counting values
            index_list = values_series.index.tolist()

           # get the values related to each of the states

           # check if D is a key
            if "D" in index_list:

                # if so , add that value in the dataframe
                output_dframe.iat[curr_row, 2] = values_series['D']
            else:

                # otherwise, set that to 0
                output_dframe.iat[curr_row, 2] = 0

            # check if H is a key
            if "H" in index_list:

                # if so, add that value in the dataframe
                output_dframe.iat[curr_row, 3] = values_series['H']
            else:

                # otherwise set that to 0
                output_dframe.iat[curr_row, 3] = 0

            # check if I is a key
            if "I" in index_list:

                # if so , add that value in the dataframe
                output_dframe.iat[curr_row, 4] = values_series['I']
            else:

                # otherwise set that to 0
                output_dframe.iat[curr_row, 4] = 0

            # check if M is a key
            if 'M' in index_list:

                # if so , add that value to the dataframe
                output_dframe.iat[curr_row, 5] = values_series['M']

            else:

                # otherwise set that to 0
                output_dframe.iat[curr_row, 5] = 0

            # check if S is a key
            if 'S' in index_list:

                # if so, add that value to the dataframe
                output_dframe.iat[curr_row, 6] = values_series['S']

            else:

                # otherwise set that to 0
                output_dframe.iat[curr_row, 6] = 0

            # update the row for filling the dataframe purpose
            curr_row += 1

        # update the date
        curr_date_df += pd.Timedelta("1 day")

    # create the csv file
    output_dframe.to_csv("a3-covid-summary-timeseries.csv", index=False)

    return output_dframe


def run(countries_csv_name, countries, start_date, end_date, sample_ratio):
    '''
        # below function will create the 2 csv file outputs
    '''

    # read the countries csv file and save it to a dataframe
    d_fr = pd.read_csv(countries_csv_name)

    # get only the given countries as the function parameter
    # filter out the given country list
    given_country_info = d_fr[d_fr['country'].isin(countries)]

    # reset the index of the new data frame
    given_country_info = given_country_info.reset_index(drop=True)

    # set the correct order of indexing
    indexing_list = []
    for curr_country in range(len(countries)):
        indexing_list.append(sorted(countries).index(countries[curr_country]))

    # set the index
    given_country_info = given_country_info.reindex(indexing_list)

    # finding the sample population
    given_country_info['population'] = (
        given_country_info['population']/sample_ratio).astype(int)

    # resetting the index
    given_country_info = given_country_info.reset_index(drop=True)

    # get the percentage
    given_country_info[given_country_info.columns[3:]] = (
        (given_country_info[given_country_info.columns[3:]].multiply(given_country_info['population'], axis="index"))/100)

    # cast as int

    given_country_info[given_country_info.columns[3:]] = (
        given_country_info[given_country_info.columns[3:]]).astype(int)

    # counting the total number of people
    people_count = 0
    for curr_country in range(3, 8):
        people_count += given_country_info[given_country_info.columns[curr_country]].sum()

    # counting the total number of days

    # first get the start date as a pandas series
    start_date_series = pd.Series(pd.to_datetime(start_date))

    # then convert the end date as a series
    end_date_series = pd.Series(pd.to_datetime(end_date))

    # then get the difference as a series
    day_count = end_date_series - start_date_series

    # then count the number of days by dividing by one day
    day_count = day_count/np.timedelta64(1, 'D')

    # then convert the date as an int
    day_count = day_count.astype(int)

    # get the days and add 1 to compensate
    day_count = day_count[0]+1

    # get the states list form the simulation parameters
    states_list = list(TRASITION_PROBS.keys())

    # get the states in the nested dictionary as well
    nested_dict_states = list(TRASITION_PROBS[states_list[0]].keys())

    # get the probabilities
    # initially it is the trainsition probability
    transition_probability = TRASITION_PROBS

    # update the dictionay

    # go through every key
    for curr_country in range(len(states_list)):

        # go through every age group
        for curr_age_group in range(len(nested_dict_states)):

            # get the probabiliyt from  the dictioanry
            transition_probability[states_list[curr_country]][nested_dict_states[curr_age_group]] = nested_dictionary(
                transition_probability[states_list[curr_country]][nested_dict_states[curr_age_group]])

    # get the hodling times
    hold_times = HOLDING_TIMES

    # holding age list
    hold_ages = list(hold_times.keys())

    # probability of ages
    age_probabilities = list(transition_probability.keys())

    # create the simulated series csv
    simulated_df = create_simulated_csv(people_count, day_count, given_country_info, start_date_series,
                                        transition_probability, age_probabilities, hold_times, hold_ages)

    # create the summary csv file
    summary_df = create_summary_to_csv(countries, day_count,
                                       start_date_series, given_country_info, simulated_df.copy())

    # then plot all the information
    create_plot('a3-covid-summary-timeseries.csv',
                ['Afghanistan', 'Sweden', 'Japan'])


if __name__ == "__main__":
    run(countries_csv_name='a3-countries.csv',
        countries=['Afghanistan', 'Sweden', 'Japan'], sample_ratio=1e6, start_date='2021-04-01', end_date='2022-04-30')
