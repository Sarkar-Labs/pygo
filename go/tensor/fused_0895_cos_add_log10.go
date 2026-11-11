package tensor

import "fmt"

func Fused0895CosAddLog10(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_log10: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Log10()
	return r, nil
}
func Fused0895CosAddLog10Name() string { return "cos_add_log10" }
