package tensor

import "fmt"

func Fused0840CosAddLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_log: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Log()
	return r, nil
}
func Fused0840CosAddLogName() string { return "cos_add_log" }
