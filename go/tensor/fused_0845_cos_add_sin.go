package tensor

import "fmt"

func Fused0845CosAddSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("cos_add_sin: nil operand")
	}
	r := x.Add(y)
	r = r.Cos()
	r = r.Sin()
	return r, nil
}
func Fused0845CosAddSinName() string { return "cos_add_sin" }
