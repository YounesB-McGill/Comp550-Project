# Comp550 Natural Language Processing Project

Fall 2019
Younes Boubekeur, Trung Vuong Thien, Anmoljeet Gill

## Introduction

This project uses [Umple](http://umple.org/), developed at the University of Ottawa, as a modeling tool and API.

## Setting up
1. Clone this repo, and follow the steps [here](https://github.com/umple/Umple/wiki/SettingUpLocalUmpleOnlineWebServer), including doing the full build. It does take a few minutes to complete, but since we won't modify Umple itself this only needs to be done once.

2. In `build/`, run this command to build the UmpleOnline website:

    ```bash
    ant -DshouldPackageUmpleOnline=true -Dmyenv=local -f build.umple.xml packageUmpleonline
    ```

3. Install these pip packages:
    ```bash
    pip3 install -U flask
    pip3 install -U flask-cors
    ```
