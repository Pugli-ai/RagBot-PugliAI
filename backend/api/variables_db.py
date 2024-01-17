OPENAI_API_KEY= ''
PINECONE_API_KEY =  ''
PINECONE_INDEX_NAME = 'pugliai-tiledesk'
eof_index = "ogg56p8zrw5wjubcmcphnwalrvbbduedcva9vdx295lpczyras"

# Delete this 2 line in production
OPENAI_API_KEY= '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'

#PINECONE_API_KEY =  '11163959-d3d5-499b-9b45-d07bd243788b'
#PINECONE_API_KEY_ZONE = 'us-west1-gcp-free'

PINECONE_API_KEY = '6d696a56-884d-4679-b357-eb429101c795' #tiledesk pinecone api
PINECONE_API_KEY_ZONE = 'eu-west4-gcp' #tiledek pinecone api zone

# List of file extensions to avoid
avoid_extensions = [
    '.pdf',  # Document files
    '.doc', '.docx',  # Microsoft Word documents
    '.ppt', '.pptx',  # Microsoft PowerPoint presentations
    '.xls', '.xlsx',  # Microsoft Excel spreadsheets
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg',  # Image files
    '.mp3', '.wav', '.ogg', '.flac',  # Audio files
    '.mp4', '.avi', '.mkv', '.flv', '.mov',  # Video files
    '.zip', '.rar', '.tar', '.gz', '.7z',  # Compressed files
    '.js',  # JavaScript files
    '.css',  # Cascading Style Sheets
    '.xml',  # XML files (unless you're interested in these)
    '.json',  # JSON files (unless you're interested in these)
    '.ico',  # Icon files
    '.swf',  # Adobe Flash files
    '.exe',  # Executable files
    '.dll',  # Dynamic Link Library files
    '.asp', '.aspx',  # Active Server Pages
    '.php',  # PHP files
    '.rss',  # RSS feed files
    '.csv'   # Comma-separated values (unless you're interested in these)
]