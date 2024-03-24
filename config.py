# Import necessary modules: Celery for task queue management, and Flask to create web applications.
from celery import Celery, Task
from flask import Flask


# Define a function to initialize Celery with a Flask application.
def celery_init_app(app: Flask) -> Celery:
    # Define a custom Celery Task class that enables tasks to use the Flask application context.
    class FlaskTask(Task):
        # Override the __call__ method to wrap task execution within the Flask application context.
        # This allows tasks to access Flask's features like database connections and application configurations.
        def __call__(self, *args: object, **kwargs: object) -> object:
            # Use Flask's app_context() to ensure tasks have access to the Flask app context.
            with app.app_context():
                # Execute the task.
                return self.run(*args, **kwargs)

    # Create a new Celery application instance named after the Flask application.
    # The task_cls parameter specifies that all tasks should use the custom FlaskTask class.
    celery_app = Celery(app.name, task_cls=FlaskTask)

    # Configure the Celery instance with the Flask application's configuration settings.
    # This typically includes broker settings, result backend, and other Celery-specific configurations.
    celery_app.config_from_object(app.config["CELERY"])

    # Make this Celery app the default app, simplifying task import and use.
    celery_app.set_default()

    # Store the Celery app instance in the Flask application's extensions dictionary.
    # This makes it accessible throughout the Flask application.
    app.extensions["celery"] = celery_app

    # Return the initialized Celery app instance.
    return celery_app
