# -*- coding: utf-8 -*-
# Generated by Django 1.9.7 on 2017-03-17 19:07
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('human_feedback_api', '0007_feedback_priority'),
    ]

    operations = [
        migrations.AddField(
            model_name='feedback',
            name='note',
            field=models.TextField(default=b'', verbose_name=b'note to be displayed along with the query'),
        ),
    ]
