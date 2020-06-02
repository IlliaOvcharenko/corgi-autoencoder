import requests
import json
import time

import pandas as pd

from fire import Fire
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image
from attrdict import AttrDict
from multiprocessing import Pool
from functools import partial


def parse_node(node, images_folder):
    info = {}
    try:
        info["shortcode"] = node.node["shortcode"]
        info["typename"] = node.node["__typename"]
        info["display_url"] = node.node["display_url"]
        info["liked_by"] = node.node["edge_liked_by"]["count"]
        info["owner_id"] = node.node["owner"]["id"]
        info["taken_at_timestamp"] = node.node["taken_at_timestamp"]
        info["accessibility_caption"] = node.node["accessibility_caption"]
        info["parsed_at_timestamp"] = datetime.now().timestamp()

        if len(node.node["edge_media_to_caption"]["edges"]) > 0:
            info["caption"] = node.node["edge_media_to_caption"]["edges"][0]["node"]["text"]

        if images_folder:
            raw = requests.get(info["display_url"])
            image = Image.open(BytesIO(raw.content))
            image.save(images_folder / (info["shortcode"] + ".jpg"), quality=95)

    except Exception as err:
        print(err)
    
    return info

def parse_hashtag(hashtag, limit=None, images_folder=None, sleep=None, verbose=True, n_proc=6):
    next_url = f"https://www.instagram.com/explore/tags/{hashtag}/?__a=1"
    has_next_page = True
    df = pd.DataFrame()
    
    try:
        while has_next_page:
            response = requests.get(next_url).text
            response = AttrDict(json.loads(response))

            has_next_page = response.graphql.hashtag.edge_hashtag_to_media.page_info.has_next_page
            end_cursor = response.graphql.hashtag.edge_hashtag_to_media.page_info.end_cursor
            next_url = f"https://www.instagram.com/explore/tags/{hashtag}/?__a=1&max_id={end_cursor}"

            nodes = response.graphql.hashtag.edge_hashtag_to_media.edges
            if len(df) + len(nodes) > limit:
                nodes = nodes[:limit - len(df)]
            
            with Pool(n_proc) as p:
                infos = p.map(partial(parse_node, images_folder=images_folder), nodes)
            
            for info in infos:
                df = df.append(info, ignore_index=True)
            
            if verbose:
                print(f"parsed images: {len(df)}")

            if limit and len(df) >= limit:
                return df

            if sleep:
                time.sleep(sleep)

        return df
    except (Exception, KeyboardInterrupt) as err:
        print(err)
        return df


def run_parser(hashtag, output_fn, limit=None, images_folder=None, sleep=None, n_proc=6):
    output_fn = Path(output_fn)
    images_folder = Path(images_folder)

    df = parse_hashtag(hashtag, limit=limit, images_folder=images_folder, sleep=sleep, n_proc=n_proc)
    df.to_csv(output_fn, index=False)
    

if __name__ == "__main__":
    Fire(run_parser)
    