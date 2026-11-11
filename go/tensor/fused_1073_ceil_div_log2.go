package tensor

import "fmt"

func Fused1073CeilDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("ceil_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Ceil()
	r = r.Log2()
	return r, nil
}
func Fused1073CeilDivLog2Name() string { return "ceil_div_log2" }
