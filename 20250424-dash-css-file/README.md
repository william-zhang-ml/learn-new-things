**How to use an external CSS file to control dashboard layout and styling**

Option 1
```
# using URLs/CDN
app = Dash(__name__, external_stylesheets=[ ... ])
```

Option 2

Write your own local `style.css` in the appropriate file location.
```
proect_folder
|-- app.py
|-- assets/
    |-- styles.css
```

**Why this is a good idea**
Good practice says that one ought to compartmentalize app layout from app styling.
With Dash, the coder declares dashboard components in the `.py` file.
This is analogous to writing an HTML file.
Dash also lets the coder style components with CSS in the `.py` file.
This is analogous to in-line CSS.
Good practice says no-no to in-line CSS and yes to external CSS.

**Journal entry?**
I've recently found an interest in Dash as a rapid-prototyping tool for data-oriented apps.
In the past I've played around with Flask, Django, and FastAPI.
I've also played around with HTML, CSS, and Javascript.
Those experiences have helped me get used to the whole backend/frontend, server/client philosophy.
What I want Dash to do for me is fill in the gap of math/graphs/etc in a web app.