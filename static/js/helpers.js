var _CHARTS = [];

function syncHandler(e) {
    for (var i = 0; i < _CHARTS.length; i++) {
        var chart = _CHARTS[i];

        if (!chart.options.axisX)
            chart.options.axisX = {};

        if (!chart.options.axisY)
            chart.options.axisY = {};

        if (e.trigger === "reset") {

            chart.options.axisX.viewportMinimum = chart.options.axisX.viewportMaximum = null;
            chart.options.axisY.viewportMinimum = chart.options.axisY.viewportMaximum = null;

            chart.render();

        } else if (chart !== e.chart) {

            chart.options.axisX.viewportMinimum = e.axisX[0].viewportMinimum;
            chart.options.axisX.viewportMaximum = e.axisX[0].viewportMaximum;

            // chart.options.axisY.viewportMinimum = e.axisY[0].viewportMinimum;
            // chart.options.axisY.viewportMaximum = e.axisY[0].viewportMaximum;

            chart.render();

        }
    }
}

