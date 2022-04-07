#!/bin/sh
git archive --remote=https://github.com/PaddlePaddle/Paddle develop:cmake/external/boost.cmake cmake/external/boost.cmake | tar -x
