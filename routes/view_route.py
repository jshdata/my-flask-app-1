from flask import render_template, request, jsonify, Blueprint
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes
import numpy as np


view_route = Blueprint('view_route', __name__)


@view_route.route("/car-damage")
def CarDamage():
    return render_template("car-damage.html")

@view_route.route("/dog-cat")
def DogCat():
    return render_template("dog-cat.html")

@view_route.route("/house")
def House():
    return render_template("house.html")

@view_route.route("/save-user")
def SaveUser():
    return render_template("save-user.html")


@view_route.route("/detail-todo")
def detailTodo():

    todo_id = request.args.get('todoid')
    
    return render_template(
            "detail-todo.html",
            todo_id=todo_id
        )


@view_route.route("/js-basic")
def jsBasic():
    return render_template("js-basic.html")

@view_route.route("/")
def home():
    return render_template("index.html")

@view_route.route("/front")
def front():
    return render_template("front.html")


@view_route.route("/login")
def gdgd():
    return render_template("login.html")


@view_route.route("/ic")
def ic():
    return render_template("id-class.html")

@view_route.route("/layout")
def layout():
    return render_template("layout.html")


if __name__ == "__main__":
    view_route.run(debug=True)
