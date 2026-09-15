# Admin and Manager Approval Rules

- `employee` requests are approved/rejected only by the manager of the employee's team.
- `manager` requests are approved/rejected only by an `admin`.
- `admin` requests cannot be approved through the current workflow.
- No user can approve or reject their own leave request.
- Only admins can create teams and assign/remove team members.
- A manager can manage only one team.
- Team membership, not department, controls employee approval authorization.

## Initial admin setup

Because the team model is already deployed and the application previously had only `employee`/`manager` roles, choose an existing trusted account and promote it directly in PostgreSQL once:

```sql
UPDATE employees
SET role = 'admin'
WHERE id = <trusted_admin_employee_id>;
```

Then restart FastAPI and use that account for team administration and manager-leave approval.
