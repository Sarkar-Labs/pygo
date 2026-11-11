package tensor

import "fmt"

func Fused0043CosDiv(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_div: nil operand")
	}
	r := x.Div(y)
	r = r.Cos()
	return r, nil
}
func Fused0043CosDivName() string { return "cos_div" }
