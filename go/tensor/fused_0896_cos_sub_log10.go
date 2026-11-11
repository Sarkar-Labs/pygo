package tensor

import "fmt"

func Fused0896CosSubLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_sub_log10: nil operand")
	}
	r := x.Sub(y)
	r = r.Cos()
	r = r.Log10()
	return r, nil
}
func Fused0896CosSubLog10Name() string { return "cos_sub_log10" }
