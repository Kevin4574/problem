from flask import Flask, render_template
from static import plot_chart
import plotly, json
import plotly.graph_objs as go

ids,figure_json = plot_chart.plot1()
print(ids)
print(figure_json)

# set up our applications
app = Flask(__name__)

# create index route
@app.route('/')
def index():
    # send the json figure to the font end
    return render_template('index.html',
                           jfile = figure_json,
                           ids = ids)

if __name__ == '__main__':
    app.run(debug=True)



