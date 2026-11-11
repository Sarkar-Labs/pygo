package tensor

import "fmt"

func Fused0660LogAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Log()
	r = r.Log()
	return r, nil
}
func Fused0660LogAddLogName() string { return "log_add_log" }
