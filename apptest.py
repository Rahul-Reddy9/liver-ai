from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
at.button[0].click().run(); assert not at.exception, at.exception
for p in ["2 · Explanation (XAI)","3 · AI Assistant","4 · Model Comparison","5 · Report"]:
    at.sidebar.radio[0].set_value(p).run(); assert not at.exception, (p, at.exception); print("ok", p)
