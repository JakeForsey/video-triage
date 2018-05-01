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
            (r"/users", UsersHandler),                      # returns available users for requested: project_id
            (r"/available_files", AvailableFilesHandler),   # returns available files for requested: project_id, user_id, file_type [video, report or image]
            (r"/upload", UploadHandler),                    # uploads a file for the reuqested: project_id, user_id and file_type
            (r"/", Userform),                               # temporary handler to test upload implementation
            (r"/process_video", ProcessVideoHandler),       # begins processing the requested video: project_id, user_id, video_id
            (r"/process_reports", ProcessReportsHandler)      # begins processing all the requested reports: project_id
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


class UsersHandler(RequestHandler):
    def get(self):
        try:
            project = self.get_argument('project_id')
            users = [name for name in os.listdir(os.path.join('projects', project)) if os.path.isdir(os.path.join('projects', project, name))]
            
            self.write(json.dumps({'users': users}))

        except Exception as e:
            self.write(json.dumps({ 'error': str(e)}))


class AvailableFilesHandler(RequestHandler):
    def get(self):
        try:
            project = self.get_argument('project_id')
            user = self.get_argument('user_id')
            file_type = self.get_argument('file_type')
            print(os.path.join('projects', project, user, file_type + 's'))
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
    def process_video(self, project, user, video):
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
                                          show=False)

        return report

    @coroutine
    def get(self):
            project = self.get_argument('project_id')
            user = self.get_argument('user_id')
            video = self.get_argument('video_id')
            report = yield self.process_video(project, user, video)

            text = '\n'.join([record[1] for record in report])

            with open(os.path.join('projects', project, user, 'reports', os.path.splitext(video)[0] + '.txt'), 'w') as f:
                for record in report:
                    f.write('{}, {}\n'.format(record[0], record[1]))

            self.write(json.dumps({'status': '{} has been processed!'.format(video)}))


class ProcessReportsHandler(RequestHandler):
    executor = ThreadPoolExecutor(max_workers=2)

    @run_on_executor
    def process_reports(self, project):
        documents = []
        for user in [name for name in os.listdir(os.path.join('projects', project)) if os.path.isdir(os.path.join('projects', project, name))]:
            reports = os.listdir(os.path.join('projects', project, user, 'reports'))

            for report in reports:
                with open(os.path.join('projects', project, user, 'reports', report), 'r') as file:
                    raw_document = ''
                    for record in file.readlines():
                        raw_document += record.split(',')[1]

                    document = []
                    for word in raw_document.split():
                        if word not in ['a', '.', 'the', 'with', 'of', 'in', 'is', 'on']:
                            document.append(word)

                documents.append(' '.join(document))

        dictionary, corpus, lda_model = categorise.build_lda_model(documents, 3)

        return pyLDAvis.prepared_data_to_html(pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False))

    @coroutine
    def get(self):
            project = self.get_argument('project_id')

            html_visualisation = yield self.process_reports(project)

            self.write(html_visualisation)


def main():
    parse_command_line()
    http_server = HTTPServer(Application())
    http_server.listen(options.port)
    IOLoop.current().start()


if __name__ == "__main__":
    main()
