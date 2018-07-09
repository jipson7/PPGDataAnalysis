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
    return chart;
}


Vue.filter('formatDate', function(value) {
    if (value) {
        return (new Date(value)).toLocaleString();
    }
});

Vue.component('chart', {
    props: ['title', 'lines'],
    mounted: function () {
        var rendered_chart = createChart(this.lines, this.title, this.$el);
        _CHARTS.push(rendered_chart);
    },
    template: `
        <div ref="container" style="height: 360px; width: 100%;"></div>
    `
});

Vue.component('led-chart', {
    props:['trialid', 'device', 'title'],
    data: function() {
        return {
            redline: [],
            irline: []
        }
    },
    created: function() {
        var self = this;
        var params = {
            data: 'red',
            device: self.device
        };
        $.get("/trials/" + self.trialid + "/chart", params, function(data) {
            if (data.data.length === 0){

            } else {
                self.redline = data.data;
            }
        });
        params.data = 'ir';
        $.get("/trials/" + self.trialid + "/chart", params, function(data) {
            self.irline = data.data;
        });
    },
    computed: {
        lines: function() {
            if ((this.redline.length > 0) && (this.irline.length > 0)) {
                return [{
                    type: "line",
                    dataPoints: this.redline,
                    legendText: "Red",
                    showInLegend: true,
                    color: "red"
                }, {
                    type: "line",
                    dataPoints: this.irline,
                    legendText: "IR",
                    showInLegend: true,
                    color: "pink"
                }]
            } else {
                return [];
            }
        }
    },
    template: `
        <div v-if="lines.length > 0">
            <chart :title="title" :lines="lines"></chart>
        </div>
    `
});

Vue.component('trial-charts', {

    props: ['trialid'],
    watch: {
        trialid: function() {
            _CHARTS = [];
        }
    },
    template: `
        <div>
            <h1 v-if="trialid === null">Select a Trial to load...</h1>
            <div v-else>
                <led-chart :trialid="trialid" 
                       :device="0"
                       :title="'Wrist Worn LED Readings'"></led-chart>
                 <led-chart :trialid="trialid" 
                       :device="1"
                       :title="'Fingertip LED Readings'"></led-chart>
            </div>
        </div>
    `
});

Vue.component('trials-list', {
    props: ['trials'],
    template: `
        <div>
        <h3>Trials</h3>
        <hr>
        <ul class="list-group">
            <li v-on:click="$emit('selected-trial', trial.id)" class="list-group-item" v-for="trial in trials">
                <span>{{trial.created | formatDate}} - {{trial.user.name}}</span>
            </li>
        </ul>
        </div>
    `
});

new Vue({
    el: "#app",
    data: function () {
        return {
            trials: [],
            trialid: null
        }
    },
    template: `
        <div class="row">
            <div class="col-sm-3">
                <trials-list v-on:selected-trial="setTrial" :trials="trials"></trials-list>
            </div>
            <div class="col-sm-9">
                <trial-charts :trialid="trialid"></trial-charts>
            </div>
        </div>
    `, methods: {
        setTrial: function (trialId) {
            this.trialid = trialId;
        }
    }, created: function () {
        console.log("Created");
        var self = this;
        $.get("/trials", function (data) {
            self.trials = data.trials;
        });
    }
});
