#!/bin/bash

# Logs all the team abbreviations. For use in matching this api's abbreviations to my configuration's.

base_url="http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2023/teams/"

current_id=1

while true; do
    response=$(curl -s "$base_url$current_id?lang=en&region=us")

    if [[ $response == *"Team not available for given id"* || $response == *'"code":400'* ]]; then
        echo "Invalid result for team ID $current_id. Exiting loop."
        break
    elif [[ $response == *"abbreviation"* ]]; then
        abbreviation=$(jq -r '.abbreviation' <<< "$response")
        echo "$abbreviation"
    else
        echo "Unknown error for team ID $current_id. Exiting loop."
        break
    fi

    ((current_id++))
done

# Logs from run @ 02/19/2024:
# ATL
# BUF
# CHI
# CIN
# CLE
# DAL
# DEN
# DET
# GB
# TEN
# IND
# KC
# LV
# LAR
# MIA
# MIN
# NE
# NO
# NYG
# NYJ
# PHI
# ARI
# PIT
# LAC
# SF
# SEA
# TB
# WSH
# CAR
# JAX
# AFC
# NFC
# BAL
# HOU
# RICE
# IRVIN
# Invalid result for team ID 37. Exiting loop.