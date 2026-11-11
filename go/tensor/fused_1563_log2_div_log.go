package tensor

import "fmt"

func Fused1563Log2DivLog(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_div_log: nil operand")
	}
	r := x.Div(y)
	r = r.Log2()
	r = r.Log()
	return r, nil
}
func Fused1563Log2DivLogName() string { return "log2_div_log" }
