class organizationDescription(AbstractBaseModel):
    title = models.CharField(max_length=255)
    company = models.CharField(max_length=255)
    location = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.title

    @classmethod
    def import_organization_descriptions(cls):
        def get_data_csv():
            """Find all the CSV files in the jobs directory and return them as a generator"""
            data_directory = os.path.join(settings.BASE_DIR, "..", "..", "data", "jobs")
            csv_paths = glob.glob(f"{data_directory}/*.csv")
            for csv_path in csv_paths:
                with open(csv_path, "r") as f:
                    yield f

        for csvfile in get_data_csv():
            print(f"Loading {csvfile.name}...")
            start_time = time.time()

            csvreader = csv.DictReader(csvfile, delimiter=",")

            job_descriptions = []
            for row in csvreader:
                organization_descriptions.append(
                    cls(
                        title=row["title"],
                        company=row["company"],
                        location=row["location"],
                        description=row["description"],
                )

            cls.objects.bulk_create(job_descriptions)

            print(f"    Loaded in {time.time() - start_time} seconds.")

    @classmethod
    def detect_location(cls):
        def get_organization_descriptions():
            page_size = 1000
            num_pages = cls.objects.count() // page_size + 1
            for page in range(num_pages):
                qs = cls.objects.filter(location="")[page * page_size : (page + 1) * page_size]
                for organization_description in qs:
                    yield organization_description

        count = 0
        total_jds = cls.objects.filter(location="").count()
        for organization_description in get_organization_descriptions():
            count += 1
            print(f"Detecting language for JD #{count} of {total_jds}...")
            try:
                language = detect(organization_description.description)
            except LangDetectException:
                language = ""
            organization_description.language = language
            organization_description.save()

    def generate_embeddings(self):
        def strip_html_tags(text):
            tag_re = re.compile(r"(<!--.*?-->|<[^>]*>)")
            no_tags = tag_re.sub("", text)
            return html.escape(no_tags)

        def get_chunks(jd_content, chunk_size=750):
            """Naive chunking of job description.

            `chunk_size` is the number of characters per chunk.
            """
            chunk_size = chunk_size
            content = strip_html_tags(jd_content).replace("\n", " ")
            while content:
                chunk, content = content[:chunk_size], content[chunk_size:]
                yield chunk

        # 1. Set up the embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # 2. Chunk the job description into sentences
        chunk_embeddings = ((c, tokenizer.tokenize(c), model.encode(c)) for c in get_chunks(self.description))

        # 3. Save the embeddings information for each chunk
        jd_chunks = []
        for chunk_content, chunk_tokens, chunk_embedding in chunk_embeddings:
            jd_chunks.append(
                organizationDescriptionChunk(
                    organization_description=self,
                    chunk=chunk_content,
                    token_count=len(chunk_tokens),
                    embedding=chunk_embedding,
                )
            )

        JobDescriptionChunk.objects.bulk_create(jd_chunks)

    @classmethod
    def search(cls, query=None):
        query = (
            query
            or "The student would prefer a job in the arts. They have a background in choir and theater. Major: Music. Minor: Theater. Graduating Year: 2022"
        )
        # > expected result: list of Job Descriptions in descending order of relevance
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(query)

        jd_chunks = organizationDescriptionChunk.objects.annotate(distance=L2Distance("embedding", query_embedding)).order_by("distance")

        unique_jds = {}
        for chunk in jd_chunks:
            if chunk.organization_description.id not in unique_jds:
                unique_jds[chunk.organization_description.id] = {
                    "organization_description": chunk.organization_description,
                    "chunks": [chunk],
                }
            else:
                unique_jds[chunk.organization_description.id]["chunks"].append(chunk)

        results = []
        for k, v in unique_jds.items():
            score = sum([c.distance for c in v["chunks"]]) / len(v["chunks"])
            job_description = v["organization_description"]
            results.append(organizationDescriptionSearchResult(score, organizationjob_description, v["chunks"]))

        return sorted(results, key=lambda r: r.score)


class organizationDescriptionChunk(AbstractBaseModel):
    organization_description = models.ForeignKey(organizationJboDescription, on_delete=models.CASCADE, related_name="chunks")
    chunk = models.TextField()
    token_count = models.IntegerField(null=True)
    embedding = VectorField(dimensions=384)

    def __str__(self):
        return f"{self.organization_description.title} - {self.chunk[:50]}"


class organizationDescriptionSearchResult:
    def __init__(self, score, job_description, chunks):
        self.score = score
        self.job_description = job_description
        self.chunks = chunks

    def __str__(self):
        return f"{self.score}: {self.job_description.title}"
