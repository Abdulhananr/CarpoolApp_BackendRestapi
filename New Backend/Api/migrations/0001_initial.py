# Generated by Django 3.2.8 on 2023-05-27 15:33

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Carpool',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lat', models.FloatField()),
                ('long', models.FloatField()),
                ('des_lat', models.FloatField()),
                ('des_long', models.FloatField()),
                ('client_id', models.IntegerField()),
                ('assien_driver', models.IntegerField(blank=True, null=True)),
                ('date', models.CharField(blank=True, max_length=24, null=True)),
                ('time', models.CharField(blank=True, max_length=24, null=True)),
                ('complete', models.CharField(blank=True, max_length=24, null=True)),
                ('price', models.FloatField()),
                ('seat', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Carpool_request',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lat', models.FloatField()),
                ('long', models.FloatField()),
                ('des_lat', models.FloatField()),
                ('des_long', models.FloatField()),
                ('client_id', models.IntegerField()),
                ('assien_driver', models.IntegerField(blank=True, null=True)),
                ('date', models.CharField(blank=True, max_length=24, null=True)),
                ('time', models.CharField(blank=True, max_length=24, null=True)),
                ('price', models.FloatField()),
                ('seat', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Client_location',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('cid', models.IntegerField()),
                ('lat', models.FloatField()),
                ('long', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='Contact',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=30)),
                ('email', models.EmailField(max_length=40)),
                ('phone', models.CharField(max_length=14)),
                ('desc', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Customer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=30)),
                ('email', models.EmailField(max_length=40, unique=True)),
                ('password', models.CharField(max_length=40)),
                ('phone', models.CharField(max_length=14, unique=True)),
                ('carplate', models.CharField(max_length=14)),
                ('carmodel', models.CharField(max_length=14)),
                ('image1', models.FileField(blank=True, null=True, upload_to='Media')),
                ('image2', models.FileField(blank=True, null=True, upload_to='Media')),
                ('image3', models.FileField(blank=True, null=True, upload_to='Media')),
                ('balance', models.IntegerField(blank=True, null=True)),
                ('trips_as_client', models.IntegerField(blank=True, null=True)),
                ('trips_as_captain', models.IntegerField(blank=True, null=True)),
                ('Profile', models.FileField(blank=True, null=True, upload_to='Media')),
                ('expo_token', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='DCarpool_request',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lat', models.FloatField()),
                ('long', models.FloatField()),
                ('des_lat', models.FloatField()),
                ('des_long', models.FloatField()),
                ('client_id', models.IntegerField()),
                ('assien_driver', models.IntegerField(blank=True, null=True)),
                ('date', models.CharField(blank=True, max_length=24, null=True)),
                ('time', models.CharField(blank=True, max_length=24, null=True)),
                ('price', models.FloatField()),
                ('seat', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Driver_location',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('did', models.IntegerField()),
                ('lat', models.FloatField()),
                ('long', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='Final_Carpool',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('lat', models.FloatField()),
                ('long', models.FloatField()),
                ('des_lat', models.FloatField()),
                ('des_long', models.FloatField()),
                ('client_id', models.IntegerField()),
                ('assien_driver', models.IntegerField(blank=True, null=True)),
                ('date', models.CharField(blank=True, max_length=24, null=True)),
                ('time', models.CharField(blank=True, max_length=24, null=True)),
                ('price', models.FloatField()),
                ('seat', models.FloatField(blank=True, null=True)),
            ],
        ),
    ]