import pandas as pd
import pandas_market_calendars as market_cal
import datetime

pd.set_option('future.no_silent_downcasting', True)

#Date format: Year-Month-Day
DATE_FORMAT ='%Y-%m-%d'
today = datetime.datetime.today()

cboe_calendar = market_cal.get_calendar('CBOE_Futures')

#Checks if input date is the third friday of the month.
#Returns True - False
def is_third_friday_of_month(input_date, friday_counter):
    parsed_current_date = datetime.datetime.strptime(input_date, DATE_FORMAT)
    current_week_num = parsed_current_date.isocalendar()[1]
    return (friday_counter == 3) and (parsed_current_date.weekday() == 4)


def is_business_day(input_date_str, input_date):
    seven_days = datetime.timedelta(days=7)
    start_date = input_date - seven_days

    start_date_str = start_date.strftime(DATE_FORMAT)
    end_date_str = input_date_str

    schedule = cboe_calendar.schedule(start_date_str, end_date_str)

    try:
        holiday_check = cboe_calendar.open_at_time(schedule, pd.Timestamp(input_date_str + ' 12:00', tz='America/Chicago'))
        return holiday_check
    except:
        None

def run_over_time_frame(start_yr):
    #Goes through the years

    futures_exp_dates = []

    thirty_days = datetime.timedelta(days=30)
    one_day = datetime.timedelta(days=1)

    for year in range(start_yr, today.year+2):
        #Goes through the months
        for month in range(1,13):
            month = "%02d" % month
            start_date = str(year) + '-' + str(month) + '-' + '01'

            friday_counter = 0

            #Goes through the days
            for day in range (1,32):
                new_string_part = "%02d" % (day)
                new_string_part = str(new_string_part)

                current_year, current_month, current_day = start_date.split('-')
                current_day = new_string_part

                current_date = '-'.join([current_year, current_month, current_day])
                parsed_current_date = datetime.datetime.strptime(current_date, DATE_FORMAT)

                if parsed_current_date.weekday() == 4:
                    friday_counter += 1

                if is_third_friday_of_month(current_date, friday_counter):

                    # If input date is a holiday, subtract one day until a business day is found.
                    while not is_business_day(current_date, parsed_current_date):
                        parsed_current_date = parsed_current_date - one_day
                        current_date = parsed_current_date.strftime(DATE_FORMAT)

                    current_date_minus_month = parsed_current_date - thirty_days

                    futures_exp_dates.append(current_date_minus_month.strftime(DATE_FORMAT))
                    # print('VIX EXPIRATION', current_date_minus_month.strftime(DATE_FORMAT))

                    break

    # Discarding first date element. It contains a date corresponding to the staring year's preceding december future expiration date.
    futures_exp_dates.pop(0)

    futures_exp_dates[futures_exp_dates.index("2024-06-19")] = "2024-06-18"

    return futures_exp_dates