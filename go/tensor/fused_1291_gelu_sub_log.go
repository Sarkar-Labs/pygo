package tensor

import "fmt"

func Fused1291GeluSubLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("gelu_sub_log: nil operand")
	}
	r := x.Sub(y)
	r = r.GELU()
	r = r.Log()
	return r, nil
}
func Fused1291GeluSubLogName() string { return "gelu_sub_log" }
