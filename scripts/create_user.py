"""创建系统用户（第一个 admin 用这个脚本，不走 HTTP 接口）。

为什么需要它：接口 /v1/auth/register 需要管理员权限，
而系统初始状态没有任何用户，所以第一个管理员必须由运维脚本创建——
这也是企业里的常规做法（初始账号由 DBA/运维初始化，不开放自助注册）。

用法：
    python scripts/create_user.py --username admin --password 'Admin@12345' --role admin --department-id 1
    python scripts/create_user.py --username alice --password 'User@12345' --role viewer --department-id 2
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from db_api import Session, User  # noqa: E402
from security import hash_password  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="创建/重置系统用户")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--role", default="admin", choices=["admin", "editor", "viewer"])
    parser.add_argument("--department-id", type=int, default=1)
    parser.add_argument("--update-password", action="store_true",
                        help="用户已存在时重置密码与角色")
    args = parser.parse_args()

    with Session() as session:
        user = session.query(User).filter(User.username == args.username).first()
        if user is not None:
            if not args.update_password:
                sys.exit(f"用户已存在: {args.username}（如需重置请加 --update-password）")
            user.password_hash = hash_password(args.password)
            user.role = args.role
            user.department_id = args.department_id
            user.is_active = True
            action = "已更新"
        else:
            session.add(User(
                username=args.username,
                password_hash=hash_password(args.password),
                role=args.role,
                department_id=args.department_id,
                is_active=True,
            ))
            action = "已创建"
        session.commit()

    print(f"{action}用户: username={args.username} role={args.role} "
          f"department_id={args.department_id}")


if __name__ == "__main__":
    main()
