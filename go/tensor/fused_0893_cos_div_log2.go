package tensor

import "fmt"

func Fused0893CosDivLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div_log2: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	r = r.Log2()
	return r, nil
}
func Fused0893CosDivLog2Name() string { return "cos_div_log2" }
