import webview
import time
import threading

def run_test():
    print("Windows at start:", webview.windows)
    w1 = webview.create_window("Window 1", html='<h1>1</h1>')
    print("Windows after w1 creation:", webview.windows)
    time.sleep(2)
    print("Windows after start:", webview.windows)

    def close_and_exit():
        time.sleep(1)
        print("Windows during:", [w.title for w in webview.windows])
        w1.destroy()
        time.sleep(1)
        print("Windows after closing:", [w.title for w in webview.windows])
        webview.windows[0].destroy()

    threading.Thread(target=close_and_exit).start()

w_main = webview.create_window("Main", html='<button onclick="pywebview.api.test()">Test</button>')
webview.start(run_test)
