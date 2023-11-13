# Changes dictionary 

import os 
import datetime
import pickle

os.chdir('/Users/janjarco/Programming/PrivateRepository/FlightDataThesisProject')

changes_dict = {
        # 'engine_add.price.override': # 2022-08-10
        #     {"date": datetime.datetime(2022, 8, 10, 7, 23),
        #     "description": 'add.price.override'},
        # 'engine_mixed.itineraries': # 2022-08-10
        #     {"date": datetime.datetime(2022, 8, 10, 11, 54), 
        #     "description": 'mixed.itineraries'},
        # 'mixed_layout_double_render_fix_engine_add.price.override_engine_mixed.itineraries':  # 2022-08-10
        #     {"date": datetime.datetime(2022, 8, 10, 7, 23),
        #     "description": "Fix double render issue"},
        'mixed_layout_double_render_fix':  # 2022-08-10
            {"date": datetime.datetime(2022, 8, 10, 7, 23),
            "description": "Fix double render issue"},
        'engine_avoid.cabin.downgrade': 
            {"date": datetime.datetime(2022, 8, 19, 9, 14),
            "description": 'avoid.cabin.downgrade'},
        'engine_hack.bagprice.override.Altea.FLX':
            {"date": datetime.datetime(2022, 9, 15, 7, 50),
            "description": 'hack.bagprice.override.Altea.FLX'},
        'engine_remove.itineraries.departure.too.close':
            {"date": datetime.datetime(2022, 9, 28, 11, 9),
            "description": 'remove.itineraries.departure.too.close'},
        'engine_immediate.payment.if.less.36.hours.departure':
            {"date": datetime.datetime(2022, 9, 26, 13, 38),
            "description": 'immediate.payment.if.less.36.hours.departure'},
        'engine_support.20ITINS':
            {"date": datetime.datetime(2022, 12, 21, 16, 23),
            "description": 'support.20ITINS'},
        'engine_add.support.unacceptable.connections':
            {"date": datetime.datetime(2023, 2, 16, 12, 55),
            "description": 'add.support.unacceptable.connections'},
        'engine.fix.reordering.after.maximum.results':
            {"date": datetime.datetime(2023, 1, 5, 11, 50),
            "description": 'fix.reordering.after.maximum.results'},
        'engine_gateway.migration':
            {"date": datetime.datetime(2022, 6, 13, 13, 44),
            "description": 'gateway.migration'},
        'engine_lhg.gateway.migration':
            {"date": datetime.datetime(2022, 7, 25, 11, 22),
            "description": 'lhg.gateway.migration'},
        'engine_bump.BFM.version':
            {"date": datetime.datetime(2022, 8, 12, 9, 29),
            "description": 'bump.BFM.version'},
        # 'payment_message_error_large': {
        #     "date": datetime.datetime(2023, 3, 1, 10, 13),
        #     "description": "Specific error message for failed payments with sums larger than 150 EUR (or equivalent)"
        # },
        # 'payment_danish_payment_banner': {
        #     "date": datetime.datetime(2023, 2, 27, 10, 48),
        #     "description": "Banner on Danish site displaying supported payment methods"
        # },
        # 'layout_double_render_fix': { # 2022-08-10
        #     "date": datetime.datetime(2022, 8, 10, 8, 9),
        #     "description": "Fix double render issue"
        # },
        'payment_mobile_support':
            {"date": datetime.datetime(2023, 2, 21, 11, 31),
            "description": 'mobile.pay.support.Denmark'},
        'payment_banner_message_error': {
            "date": datetime.datetime(2023, 2, 27, 10, 48),
            "description": "Banner on Danish site displaying supported payment methods"
        },
        'payment_layout_cc_error_msg': {
            "date": datetime.datetime(2023, 1, 20, 14, 23),
            "description": "More specific error messages for failing credit card payments (better reason why payment failed)"
        },
        'layout_color_change': {
            "date": datetime.datetime(2022, 12, 9, 15, 10),
            "description": "Small color change"
        },
        'layout_contact_info': {
            "date": datetime.datetime(2022, 11, 28, 8, 11),
            "description": "Extended company contact information on booking step"
        },
        'layout_child_age_cat': {
            "date": datetime.datetime(2022, 11, 17, 10, 48),
            "description": "Child search changed from using age number to age categories (infant / child)"
        },
        'layout_booking_popunder': {
            "date": datetime.datetime(2022, 11, 10, 10, 16),
            "description": "Booking.com popunder window/tab"
        },
        'layout_itinerary_bug_fix': {
            "date": datetime.datetime(2022, 7, 5, 15, 25),
            "description": "Fix bug hiding itinerary on step 4"
        },
        'layout_momondo_carryon_fix': {
            "date": datetime.datetime(2022, 6, 23, 16, 27),
            "description": "Fix Momondo carryon count"
        },
        'payment_remove_resurs_bank': {
            "date": datetime.datetime(2022, 5, 30, 10, 35),
            "description": "Remove Resurs Bank payment option in Denmark"
        },
        'layout_remove_luckyorange': {
            "date": datetime.datetime(2022, 5, 9, 13, 18),
            "description": "Remove Luckyorange and Google maps stuff to make page faster"
        },
        'placebo_1': {
            "date": datetime.datetime(2022, 6, 12, 9, 0),
            "description": "Fixed broken links on about us page"
        },
        'placebo_2': {
            "date": datetime.datetime(2022, 7, 27, 13, 0),
            "description": "Added new customer review section to product pages"
        },
        'placebo_3': {
            "date": datetime.datetime(2022, 9, 8, 16, 0),
            "description": "Updated booking confirmation email template"
        },
        'placebo_4': {
            "date": datetime.datetime(2022, 9, 24, 10, 0),
            "description": "Improved search algorithm for better results"
        },
        'placebo_5': {
            "date": datetime.datetime(2022, 10, 30, 15, 0),
            "description": "Implemented dark mode feature for website"
        },
        'placebo_6': {
            "date": datetime.datetime(2022, 12, 25, 14, 0),
            "description": "Updated privacy policy link in footer"
        },
        'placebo_7': {
            "date": datetime.datetime(2023, 1, 6, 9, 0),
            "description": "Added new FAQ section to support page"
        },
        'placebo_8': {
            "date": datetime.datetime(2023, 2, 2, 14, 0),
            "description": "Implemented auto-fill feature for booking form"
        },



        
    }

# save to pickle file
with open('data/raw/changes_dict.pickle', 'wb') as handle:
    pickle.dump(changes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

