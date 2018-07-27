Vue.filter('formatDate', function(value) {
    if (value) {
        return (new Date(value)).toLocaleString();
    }
});

(function getTrials() {
    $.get("/trials", function(data) {
        console.log(data);
        new Vue({
            el: '#trialList',
            data: {
                trials: data.trials
            },
            methods: {
                fetch: getTrial
            }
        });
    });
})();

var devices = {
    WRIST_DEVICE: 0,
    REFLECTIVE_FINGERTIP: 1,
    TRANSITIVE_FINGERTIP: 2
}

function getTrial(trialId) {
    $.get("/trials/" + trialId + "/data", function(data) {
        console.log(data);
        //chartDevices(obj.devices);
    });
}

function datifyTimestamps(data) {
    for (var key in data) {
        if (key == "_id") {
            continue;
        }
        if (data.hasOwnProperty(key)) {
            data[key].map(v => {
                v.x = new Date(v.x);
                return v;
            });
        }
    }
}

function chartDevices(devices) {
    var charts = [];

    for (var i = 0; i < devices.length; i++) {
        var device = devices[i];
        var data = device.data;
        datifyTimestamps(data);
        if (device.type == WRIST_DEVICE) {
            var led_lines = createLEDLines(data);
            var algos_lines = createAlgosLines(data);
            charts.push(createChart(led_lines, "Wrist LEDs", "wrist_device"));
            charts.push(createChart(algos_lines, "Wrist Algorithm Results", "wrist_device_algos"));
        } else if (device.type == GROUND_TRUTH) {
            datifyTimestamps(data[GROUND_TRUTH]);
            var lines = [{
                type: "line",
                dataPoints: data.hr,
                legendText: "HR",
                showInLegend: true,
                color: "red"
            }, {
                type: "line",
                dataPoints: data.oxygen,
                legendText: "Oxygen",
                showInLegend: true,
                color: "blue"
            }];
            charts.push(createChart(lines, "GROUND TRUTH", "ground_truth"));
        } else if (device.type == FINGERTIP_SENSOR) {
            datifyTimestamps(data[FINGERTIP_SENSOR]);
            var lines = createLEDLines(data);
            var algos_lines = createAlgosLines(data);
            charts.push(createChart(lines, "Fingertip LEDs", "fingertip_sensor"));
            charts.push(createChart(algos_lines, "Fingertip Algorithm Results", "fingertip_sensor_algos"));
        } else {
            console.error("Unknown device type");
        }
    }

    _CHARTS = charts;
}

function createChart(data, title, containerId) {
    var chart = new CanvasJS.Chart(containerId, {
        zoomEnabled: true,
        zoomType: "xy", // change it to "xy" to enable zooming on both axes
        title: {
            text: title
        },
        data: data,
        axisY: {
            includeZero: false
        },
        rangeChanged: syncHandler
    });
    chart.render();
    _CHARTS.push(chart);
    return chart;
}

function createLEDLines(data) {
    var led_lines = [{
        type: "line",
        dataPoints: data.red,
        legendText: "RED",
        showInLegend: true,
        color: "red"
    }, {
        type: "line",
        dataPoints: data.ir,
        lineColor: "blue",
        legendText: "IR",
        showInLegend: true,
        color: "blue"
    }];
    return led_lines;
}

function createAlgosLines(data) {
    var maxim_hr = [];
    var enhanced_hr = [];
    var maxim_oxygen = [];
    var enhanced_oxygen = [];
    for (var i = 0; i < data.algos.length; i++) {
        var pair = data.algos[i];
        var x = pair.x;
        var y_maxim = pair.y.maxim;
        var hr_maxim = (y_maxim.hr == -999) ? 0 : y_maxim.hr;
        var hr_valid_maxim = (y_maxim.hr_valid == 1) ? "Valid" : "Invalid";
        var oxygen_maxim = (y_maxim.oxygen == -999) ? 0 : y_maxim.oxygen;
        var oxygen_valid_maxim = (y_maxim.oxygen_valid == 1) ? "Valid" : "Invalid";
        var y_enhanced = pair.y.enhanced;
        var hr_enhanced = (y_enhanced.hr == -999) ? 0 : y_enhanced.hr;
        var hr_valid_enhanced = (y_enhanced.hr_valid == 1) ? "Valid" : "Invalid";
        var oxygen_enhanced = (y_enhanced.oxygen == -999) ? 0 : y_enhanced.oxygen;
        var oxygen_valid_enhanced = (y_enhanced.oxygen_valid == 1) ? "Valid" : "Invalid";
        maxim_hr.push({
            x: x,
            y: hr_maxim,
            label: hr_valid_maxim
        });
        enhanced_hr.push({
            x: x,
            y: hr_enhanced,
            label: hr_valid_enhanced
        });
        maxim_oxygen.push({
            x: x,
            y: oxygen_maxim,
            label: oxygen_valid_maxim
        });
        enhanced_oxygen.push({
            x: x,
            y: oxygen_enhanced,
            label: oxygen_valid_enhanced
        });
    }
    var lines = [{
        type: "line",
        lineDashType: "dash",
        dataPoints: maxim_hr,
        legendText: "Maxim - HR",
        showInLegend: true,
        color: "red"
    }, {
        type: "line",
        lineDashType: "dash",
        dataPoints: enhanced_hr,
        legendText: "Enhanced - HR",
        showInLegend: true,
        color: "pink"
    }, {
        type: "line",
        lineDashType: "dash",
        dataPoints: maxim_oxygen,
        legendText: "Maxim - Oxygen",
        showInLegend: true,
        color: "blue"
    }, {
        type: "line",
        lineDashType: "dash",
        dataPoints: enhanced_oxygen,
        legendText: "Enhanced - Oxygen",
        showInLegend: true,
        color: "green"
    }];
    return lines;
}
