import logging
import json
from PIL import Image
import os
import uuid
import io
from concurrent.futures import ThreadPoolExecutor

from moviepy.editor import VideoFileClip

from gensim.summarization.summarizer import summarize

import pyLDAvis.gensim
import pyLDAvis

from tornado.httpserver import HTTPServer
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from tornado.concurrent import run_on_executor
from tornado.gen import coroutine
from tornado.options import define, options, parse_command_line

from lib import caption
from lib import categorise
from lib.model import Vocabulary

from config import Config


define("port", default=8888, help="run on the given port", type=int)
define("debug", default=False, help="run in debug mode")


class Application(Application):
    def __init__(self):
        handlers = [
            (r"/projects", ProjectsHandler),                # returns available projects
            (r"/create_project", CreateProjectHandler),     # creates a project folder structure given a: project_id
            (r"/users", UsersHandler),                      # returns available users for requested: project_id
            (r"/create_user", CreateUserHandler),           # creates a user folder structure given a: project_id, user_id
            (r"/available_files", AvailableFilesHandler),   # returns available files for requested: project_id, user_id, file_type [video, report]
            (r"/upload", UploadHandler),                    # uploads a file for the requested: project_id, user_id and file_type
            (r"/", Userform),                               # temporary index to test upload implementation
            (r"/process_video", ProcessVideoHandler),       # begins processing the requested video: project_id, user_id, video_id (if save='true', captioned images will be saved)
            (r"/process_reports", ProcessReportsHandler)    # begins processing all the requested reports: project_id
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret = 'agjbagijabhpaijha',
            debug=True
            )

        super(Application, self).__init__(handlers, **settings)


class ProjectsHandler(RequestHandler):
    def get(self):
        projects = [name for name in os.listdir('projects') if os.path.isdir(os.path.join('projects', name))]

        self.write(json.dumps({'projects': projects}))


class CreateProjectHandler(RequestHandler):
    def get(self):
        project = self.get_argument('project_id')

        if os.path.isdir(os.path.join('projects', project_id)):

            self.write(json.dumps({'response': 'Project already exists'}))

        else:
            os.path.mkdir(os.path.join('projects', project_id))

            self.write(json.dumps({'response': 'Project created'}))


class UsersHandler(RequestHandler):
    def get(self):
        try:
            project = self.get_argument('project_id')
            users = [name for name in os.listdir(os.path.join('projects', project)) if os.path.isdir(os.path.join('projects', project, name))]
            
            self.write(json.dumps({'users': users}))

        except Exception as e:
            self.write(json.dumps({'error': str(e)}))


class CreateUserHandler(RequestHandler):
    def get(self):
        project = self.get_argument('project_id')
        user = self.get_argument('user_id')

        if os.path.isdir(os.path.join('projects', project, user)):

            self.write(json.dumps({'response': 'User already exists'}))

        else:
            os.mkdir(os.path.join('projects', project, user))
            os.mkdir(os.path.join('projects', project, user, 'reports'))
            os.mkdir(os.path.join('projects', project, user, 'videos'))
            os.mkdir(os.path.join('projects', project, user, 'images'))

            self.write(json.dumps({'response': 'User created'}))


class AvailableFilesHandler(RequestHandler):
    def get(self):
        try:
            project = self.get_argument('project_id')
            user = self.get_argument('user_id')
            file_type = self.get_argument('file_type')

            files = os.listdir(os.path.join('projects', project, user, file_type + 's'))

            self.write(json.dumps({file_type: files}))

        except Exception as e:
            self.write(json.dumps({ 'error': str(e)}))


# temporary handler for checking upload functionality
class Userform(RequestHandler):
    def get(self):

        self.render("fileuploadform.html")


class UploadHandler(RequestHandler):
    def post(self):
        try:
            project = self.get_argument('project_id')
            user = self.get_argument('user_id')
            file_type = self.get_argument('file_type')

            file_name = self.request.files['file'][0]['filename']
            file_extension = os.path.splitext(file_name)[1]
            file_uuid = str(uuid.uuid4()) + file_extension

            with open(os.path.join('projects', project, user, file_type + 's', file_uuid), 'wb') as file:
                file.write(self.request.files['file'][0]['body'])

            self.finish(json.dumps({'status': '{} has been uploaded!'.format(file_name)}))

        except Exception as e:
            self.write(json.dumps({ 'error': str(e)}))


class ProcessVideoHandler(RequestHandler):
    executor = ThreadPoolExecutor(max_workers=1)

    @run_on_executor
    def process_video(self, project, user, video, save):
        video = VideoFileClip(os.path.join('projects', project, user, 'videos', video))

        encoder, decoder, vocab, transform = caption.load_model(vocab_path=Config.VOCAB_PATH, 
                                                                embed_size=Config.EMBED_SIZE, 
                                                                hidden_size=Config.HIDDEN_SIZE, 
                                                                num_layers=Config.NUM_LAYERS,
                                                                encoder_path=Config.ENCODER_PATH,
                                                                decoder_path=Config.DECODER_PATH)
        
        report = caption.caption_video(encoder=encoder,
                                          decoder=decoder,
                                          vocab=vocab,
                                          transform=transform,
                                          video=video,
                                          fps=0.1,
                                          save=save,
                                          image_dir=os.path.join('projects', project, user, 'images'))

        return report

    @coroutine
    def get(self):
        project = self.get_argument('project_id')
        user = self.get_argument('user_id')
        video = self.get_argument('video_id')
        save = self.get_argument('save')

        report = yield self.process_video(project, user, video, save)

        with open(os.path.join('projects', project, user, 'reports', os.path.splitext(video)[0] + '.txt'), 'w') as f:
            for record in report:
                f.write('{}, {}\n'.format(record[0], record[1]))

        self.write(json.dumps({'status': '{} has been processed!'.format(video)}))


class ProcessReportsHandler(RequestHandler):
    executor = ThreadPoolExecutor(max_workers=2)

    @run_on_executor
    def process_reports(self, project, n_topics):
        documents = []

        for root, directories, files in os.walk(os.path.join('projects', project)):
            for file in files:

                if file.endswith('.txt'):
                    with open(os.path.join(root, file), 'r') as report_file:
                        words = []
                        for line in report_file.readlines():

                            words.extend(word for word in line.split(',')[1].split() 
                                                        if word not in Config.STOP_WORDS)

                    documents.append(' '.join(words))

        dictionary, corpus, lda_model = categorise.build_lda_model(documents, n_topics)

        return pyLDAvis.prepared_data_to_html(pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False))

    @coroutine
    def get(self):
        project = self.get_argument('project_id')
        n_topics = self.get_argument('n_topics')

        html_visualisation = yield self.process_reports(project, n_topics)

        self.write(html_visualisation)


def main():
    parse_command_line()
    http_server = HTTPServer(Application())
    http_server.listen(options.port)
    IOLoop.current().start()


if __name__ == "__main__":
    main()
