#!/usr/bin/env bash
rm -rf build/html/
make html
rsync -r build/html/ $ZISSOU_USER@zissou.infosci.cornell.edu:/var/www/html/socialkit/documentation