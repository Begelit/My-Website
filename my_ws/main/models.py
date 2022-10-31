from django.db import models
'''
class User(models.Model):
    user_email = models.CharField(max_length=100)
    user_pswd = models.CharField(max_length=100)
    user_access = models.CharField(max_length=30)
    def __str__(self):
        return self.user_telephone_number
# Create your models here.

class Profile(models.Model):
    profile_user_fk = models.ForeignKey(User,on_delete=models.CASCADE)
    profile_name = models.CharField(max_length=40)
    profile_gender = models.BooleanField()
    profile_bithDate = models.DateField()
    profile_description = models.CharField(max_length=300, blank=True,null=True)
    def __str__(self):
        return self.user_telephone_number
'''

