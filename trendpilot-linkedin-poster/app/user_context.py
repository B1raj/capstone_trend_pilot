# app/user_context.py

class UserContext:
    """
    Represents an authenticated LinkedIn user.
    """

    def __init__(self, user_id, access_token, person_urn):
        self.user_id = user_id
        self.access_token = access_token
        self.person_urn = person_urn