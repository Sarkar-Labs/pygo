package tensor

import "fmt"

func Fused0841CosSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Log()
	return r, nil
}
func Fused0841CosSubLogName() string { return "cos_sub_log" }
