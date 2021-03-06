import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def execute(model, data, savepath, *args, **kwargs):
    data.remove_all_filters()
    data.add_exclusive_filter("Data Set code", '=', 2)
    model.fit(data.get_x_data(), data.get_y_data())

    atr1_alloys = [6,34,35,36,37,38]
    data.remove_all_filters()
    data.add_inclusive_filter("Data Set code", '=', 2)

    Ypredict = model.predict(data.get_x_data())
    overall_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    #plt.scatter(data.get_y_data(), Ypredict, lw=0, color='#009AFF')
    #print(len(data.get_y_data()))

    data.remove_all_filters()
    for x in atr1_alloys:
        data.add_inclusive_filter("Alloy", '=', x)
    data.add_exclusive_filter("Data Set code", '<>', 2)
    Ypredict = model.predict(data.get_x_data())
    atr1_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    plt.scatter(data.get_y_data(), Ypredict, lw = 0, color = '#03FF00', label = 'IVAR, ATR1, ATR2')
    #print(len(data.get_y_data()))

    data.remove_all_filters()
    data.add_inclusive_filter("Alloy",'<',60)
    for x in atr1_alloys:
        data.add_exclusive_filter("Alloy", '=', x)
    data.add_exclusive_filter("Data Set code", '<>', 2)
    Ypredict = model.predict(data.get_x_data())
    ivar_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    plt.scatter(data.get_y_data(), Ypredict, lw=0, color='#FF0034', label = 'IVAR, ATR2')
    #print(len(data.get_y_data()))

    data.remove_all_filters()
    data.add_inclusive_filter("Alloy", '>', 59)
    data.add_exclusive_filter("Data Set code", '<>', 2)
    Ypredict = model.predict(data.get_x_data())
    atr2_rms = np.sqrt(mean_squared_error(data.get_y_data(), Ypredict))
    plt.scatter(data.get_y_data(), Ypredict, lw=0, color='#49CCFF', label = 'ATR2 only')
    #print(len(data.get_y_data()))

    plt.title('Predicting ATR-2, Training IVAR, ATR1')
    plt.figtext(.15, .84, 'Overall RMSE: {:.3f}'.format(overall_rms))
    plt.figtext(.15, .80, 'IVAR, ATR1, ATR2 alloys: {:.3f}'.format(atr1_rms))
    plt.figtext(.15, .76, 'IVAR, ATR2 alloys: {:.3f}'.format(ivar_rms))
    plt.figtext(.15, .72, 'ATR2 only alloys: {:.3f}'.format(atr2_rms))
    plt.legend(loc= 'lower right')
    plt.plot(plt.gca().get_ylim(), plt.gca().get_ylim(), ls="--", c=".3")
    plt.xlabel('Measured (MPa)')
    plt.ylabel('Predicted (MPa)')
    plt.savefig(savepath.format(plt.gca().get_title()), dpi=300, bbox_inches='tight')
    plt.close()
